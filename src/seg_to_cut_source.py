"""Build synthetic recipes from completed agir-pipeline seg_to_cut runs."""

import json
import logging
import math
import random
import uuid
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _artifact_path(run_dir, item, name):
    relative = item["artifacts"][name]
    root = (run_dir / "artifacts").resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid {name}: {path}")
    return path


def _area(metadata, settings):
    props = metadata["cutout_props"]
    value = props.get("bbox_area_cm2")
    if value is None:
        value = props.get("estimated_bbox_area_cm2")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _selected(metadata, settings):
    if settings.validated is not None and metadata.get("validated") != settings.validated:
        return False
    props = metadata["cutout_props"]
    if settings.is_primary is not None and props.get("is_primary") != settings.is_primary:
        return False
    name = str(metadata["category"].get("common_name") or "").strip().lower()
    names = {value.strip().lower() for value in settings.common_names}
    if names and name not in names:
        return False
    placement = props.get("edge_cut", {}).get("synthetic_placement", "unrestricted")
    if placement not in settings.placements:
        return False
    area = _area(metadata, settings)
    if area is None or area < settings.min_area_cm2:
        return False
    if settings.max_area_cm2 is not None and area > settings.max_area_cm2:
        return False
    return True


def load_cutouts(settings):
    """Read successful manifest items without relying on a SQLite inventory."""
    if not settings.runs:
        raise ValueError("seg_to_cut.runs must contain at least one completed run directory")
    documents = []
    seen = set()
    class_names = {}
    for configured_run in settings.runs:
        run_dir = Path(configured_run).expanduser().resolve()
        manifest = _read_json(run_dir / "manifest.json")
        report = _read_json(run_dir / "run_report.json")
        if manifest.get("stage") != "seg_to_cut" or report.get("stage") != "seg_to_cut":
            raise ValueError(f"Not a seg_to_cut run: {run_dir}")
        if report.get("status") not in ("success", "partial"):
            raise ValueError(f"Run did not complete successfully: {run_dir}")
        if manifest.get("run_id") != report.get("run_id") or manifest.get("batch_id") != report.get("batch_id"):
            raise ValueError(f"Run report and manifest disagree: {run_dir}")
        for item in manifest["items"]:
            if item.get("status") != "ok":
                continue
            metadata_path = _artifact_path(run_dir, item, "metadata_path")
            png_path = _artifact_path(run_dir, item, "cutout_path")
            mask_path = _artifact_path(run_dir, item, "mask_path")
            metadata = _read_json(metadata_path)
            cutout_id = metadata.get("cutout_id")
            if cutout_id != item.get("image_id") or metadata.get("batch_id") != manifest.get("batch_id"):
                raise ValueError(f"Cutout identity mismatch: {metadata_path}")
            if cutout_id in seen:
                raise ValueError(f"Duplicate cutout_id across source runs: {cutout_id}")
            seen.add(cutout_id)
            if not _selected(metadata, settings):
                continue
            category = metadata["category"]
            class_id = category.get("cultivar_class_id") or category.get("class_id")
            if isinstance(class_id, bool) or not isinstance(class_id, int) or not 1 <= class_id <= 255:
                raise ValueError(f"Invalid class ID for {cutout_id}: {class_id}")
            class_name = str(
                category.get("cultivar_name") or category.get("common_name") or ""
            ).strip().lower()
            if not class_name:
                raise ValueError(f"Missing class name for {cutout_id}")
            if class_id in class_names and class_names[class_id] != class_name:
                raise ValueError(
                    f"Class ID {class_id} maps to both {class_names[class_id]} and {class_name}"
                )
            class_names[class_id] = class_name
            document = dict(metadata)
            document["_id"] = cutout_id
            document["cutout_path"] = str(png_path)
            document["mask_path"] = str(mask_path)
            document["source_run_id"] = manifest["run_id"]
            document["source_manifest"] = str(run_dir / "manifest.json")
            document["resolved_class_id"] = class_id
            document["synthesis_area_cm2"] = _area(metadata, settings)
            documents.append(document)
    if not documents:
        raise ValueError("No cutouts match seg_to_cut filters")
    log.info("Selected %d seg_to_cut cutouts", len(documents))
    return documents


def _sample(documents, count, weights, reuse):
    if not weights:
        return random.choices(documents, k=count) if reuse else random.sample(documents, min(count, len(documents)))
    normalized = {name.strip().lower(): float(weight) for name, weight in weights.items()}
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("seg_to_cut.class_weights cannot be negative")
    values = [normalized.get(str(doc["category"].get("common_name") or "").strip().lower(), 0) for doc in documents]
    if not any(values):
        raise ValueError("seg_to_cut.class_weights match no selected cutouts")
    if reuse:
        return random.choices(documents, weights=values, k=count)
    chosen = []
    remaining = list(zip(documents, values))
    for _ in range(min(count, len(documents))):
        positive = [(doc, weight) for doc, weight in remaining if weight > 0]
        if not positive:
            break
        selected = random.choices([doc for doc, _ in positive], weights=[weight for _, weight in positive])[0]
        chosen.append(selected)
        remaining = [(doc, weight) for doc, weight in remaining if doc["_id"] != selected["_id"]]
    return chosen


def create_recipes_from_runs(cfg: DictConfig):
    documents = load_cutouts(cfg.seg_to_cut)
    backgrounds = sorted(
        path for path in Path(cfg.paths.backgrounddir).iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
    ) if Path(cfg.paths.backgrounddir).is_dir() else []
    if not backgrounds:
        raise ValueError(f"No background JPEGs in {cfg.paths.backgrounddir}")
    count_range = cfg.cutout_filters.cuts_n_image
    if count_range.min < 1 or count_range.max < count_range.min:
        raise ValueError("cuts_n_image must have 1 <= min <= max")
    if cfg.cutout_filters.total_images < 1:
        raise ValueError("total_images must be positive")
    reuse = cfg.cutout_filters.reuse_cutouts
    available = list(documents)
    recipes = []
    for _ in range(cfg.cutout_filters.total_images):
        if not available:
            break
        selected = _sample(
            available, random.randint(count_range.min, count_range.max),
            cfg.seg_to_cut.class_weights, reuse,
        )
        if not selected:
            raise ValueError("No cutouts can be sampled with the configured class weights")
        recipes.append({
            "synthetic_image_id": str(uuid.uuid4()),
            "background_image_id": random.choice(backgrounds).name,
            "cutouts": selected,
        })
        if not reuse:
            used = {doc["_id"] for doc in selected}
            available = [doc for doc in available if doc["_id"] not in used]
    output_dir = Path(cfg.paths.recipesdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{cfg.project_name}_{cfg.sub_name}.json"
    output.write_text(json.dumps({"synthetic_images": recipes}, indent=2), encoding="utf-8")
    log.info("Saved %d recipes to %s", len(recipes), output)
