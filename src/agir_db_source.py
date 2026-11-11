"""Build recipes from the flat ``semif`` table in AgIR DB v2."""

import json
import logging
import math
import random
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

PLACEMENTS = {
    "unrestricted",
    "left_edge_only", "right_edge_only", "top_edge_only", "bottom_edge_only",
    "top_left_corner_only", "top_right_corner_only",
    "bottom_left_corner_only", "bottom_right_corner_only",
}


def _mounted_path(root: Path, stored_path: str | None) -> Path | None:
    """Resolve a DB path rooted at /semifield-cutouts under a local mount."""
    if not stored_path:
        return None
    parts = PurePosixPath(stored_path).parts
    if parts and parts[0] == "/":
        parts = parts[1:]
    if not parts or parts[0] != "semifield-cutouts" or ".." in parts:
        return None
    path = (root.joinpath(*parts)).resolve()
    return path if path.is_relative_to(root) else None


def _area(row: sqlite3.Row) -> float | None:
    for key in ("bbox_area_cm2", "estimated_bbox_area_cm2"):
        value = row[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value) and value > 0:
                return float(value)
    return None


def _placement(row: sqlite3.Row, metadata: dict) -> str | None:
    props = metadata.get("cutout_props") or {}
    if not isinstance(props, dict):
        return None
    edge = props.get("edge_cut") or {}
    if isinstance(edge, dict) and edge.get("synthetic_placement") is not None:
        value = edge["synthetic_placement"]
        return value if isinstance(value, str) and value in PLACEMENTS else None
    border = row["extends_border"]
    if border is None:
        border = props.get("extends_border")
    return "unrestricted" if border is False or border == 0 else None


def load_cutouts(cfg) -> list[dict]:
    """Select usable cutouts using DB columns before reading any cutout JSON."""
    names = {str(name).strip().lower() for name in cfg.cutout_filters.category.common_name if str(name).strip()}
    if not names:
        raise ValueError("agir_db requires cutout_filters.category.common_name")
    root = Path(cfg.paths.cutout_storage_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Cutout storage root does not exist: {root}")
    db_path = Path(cfg.paths.sql_database).expanduser().resolve()
    if not db_path.is_file():
        raise ValueError(f"AgIR database does not exist: {db_path}")
    placeholders = ", ".join("?" for _ in names)
    query = f"""
        SELECT cutout_id, batch_id, category_common_name, category_class_id,
               bbox_area_cm2, estimated_bbox_area_cm2, extends_border,
               cutout_path, cutout_json_path
        FROM semif
        WHERE LOWER(TRIM(category_common_name)) IN ({placeholders})
          AND category_class_id BETWEEN 1 AND 255
          AND cutout_id IS NOT NULL AND cutout_path IS NOT NULL
          AND (bbox_area_cm2 > 0 OR estimated_bbox_area_cm2 > 0)
    """
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, sorted(names)).fetchall()
    random.shuffle(rows)

    count_range = cfg.cutout_filters.cuts_n_image
    target = cfg.cutout_filters.total_images * count_range.max
    selected = []
    rejected = Counter()
    seen = set()
    for row in rows:
        if len(selected) >= target:
            break
        cutout_id = row["cutout_id"]
        if cutout_id in seen:
            rejected["duplicate ID"] += 1
            continue
        seen.add(cutout_id)
        area = _area(row)
        if area is None or area < cfg.agir_db.min_area_cm2 or (
            cfg.agir_db.max_area_cm2 is not None and area > cfg.agir_db.max_area_cm2
        ):
            rejected["area"] += 1
            continue
        png = _mounted_path(root, row["cutout_path"])
        if png is None or not png.is_file():
            rejected["missing PNG"] += 1
            continue
        metadata = {}
        json_path = _mounted_path(root, row["cutout_json_path"])
        if json_path is not None and json_path.is_file():
            try:
                with json_path.open(encoding="utf-8") as stream:
                    metadata = json.load(stream)
                if not isinstance(metadata, dict) or (
                    metadata.get("cutout_id") not in (None, cutout_id)
                ):
                    rejected["invalid JSON identity"] += 1
                    continue
            except (OSError, ValueError, TypeError):
                rejected["unreadable JSON"] += 1
                continue
        placement = _placement(row, metadata)
        if placement is None:
            rejected["unknown or unsuitable placement"] += 1
            continue
        selected.append({
            "_id": cutout_id,
            "cutout_id": cutout_id,
            "batch_id": row["batch_id"],
            "cutout_path": str(png),
            "cutout_json_path": str(json_path) if json_path and json_path.is_file() else None,
            "category": {
                "class_id": row["category_class_id"],
                "common_name": row["category_common_name"],
            },
            "resolved_class_id": row["category_class_id"],
            "synthesis_area_cm2": area,
            "synthetic_placement": placement,
        })
    log.info("AgIR DB: %d candidate rows, %d usable cutouts, rejected %s", len(rows), len(selected), dict(rejected))
    if not selected:
        raise ValueError("No AgIR DB cutouts have usable metadata, placement, and PNG files")
    return selected


def _sample(documents: list[dict], count: int, weights: dict, reuse: bool) -> list[dict]:
    if not weights:
        return random.choices(documents, k=count) if reuse else random.sample(documents, min(count, len(documents)))
    normalized = {str(name).strip().lower(): float(weight) for name, weight in weights.items()}
    if any(not math.isfinite(weight) or weight < 0 for weight in normalized.values()):
        raise ValueError("agir_db.class_weights must be finite and nonnegative")
    eligible = [doc for doc in documents if normalized.get(doc["category"]["common_name"].strip().lower(), 0) > 0]
    if not eligible:
        raise ValueError("agir_db.class_weights match no available cutouts")
    chosen = []
    remaining = eligible.copy()
    for _ in range(count if reuse else min(count, len(remaining))):
        class_names = sorted({doc["category"]["common_name"].strip().lower() for doc in remaining})
        name = random.choices(class_names, weights=[normalized[n] for n in class_names])[0]
        candidates = [doc for doc in remaining if doc["category"]["common_name"].strip().lower() == name]
        doc = random.choice(candidates)
        chosen.append(doc)
        if not reuse:
            remaining.remove(doc)
    return chosen


def create_recipes_from_db(cfg) -> None:
    count_range = cfg.cutout_filters.cuts_n_image
    if count_range.min < 1 or count_range.max < count_range.min:
        raise ValueError("cuts_n_image must have 1 <= min <= max")
    if cfg.cutout_filters.total_images < 1:
        raise ValueError("total_images must be positive")
    documents = load_cutouts(cfg)
    background_dir = Path(cfg.paths.backgrounddir)
    backgrounds = sorted(
        path for path in background_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
    ) if background_dir.is_dir() else []
    if not backgrounds:
        raise ValueError(f"No background JPEGs in {background_dir}")
    recipes = []
    available = documents.copy()
    for _ in range(cfg.cutout_filters.total_images):
        if not available:
            break
        chosen = _sample(
            available, random.randint(count_range.min, count_range.max),
            cfg.agir_db.class_weights, cfg.cutout_filters.reuse_cutouts,
        )
        if not chosen:
            break
        recipes.append({
            "synthetic_image_id": str(uuid.uuid4()),
            "background_image_id": random.choice(backgrounds).name,
            "cutouts": chosen,
        })
        if not cfg.cutout_filters.reuse_cutouts:
            used = {doc["_id"] for doc in chosen}
            available = [doc for doc in available if doc["_id"] not in used]
    output_dir = Path(cfg.paths.recipesdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{cfg.project_name}_{cfg.sub_name}.json"
    output.write_text(json.dumps({"synthetic_images": recipes}, indent=2), encoding="utf-8")
    log.info("Saved %d AgIR DB recipes to %s", len(recipes), output)
