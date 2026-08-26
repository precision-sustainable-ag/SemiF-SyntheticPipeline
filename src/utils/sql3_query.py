import logging
import sqlite3
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from utils.datetime_utils import parse_datetime, percentile

log = logging.getLogger(__name__)


class SpeciesFilterQueryEngine:
    """
    Builds and executes per-species filtered queries against the cutouts
    table, driven by `cutout_filters.category.species_filters` in
    conf/cutout_filters/default.yaml.

    Unlike a single combined query for every species, this loops per
    species and, within a species, per season -- because the `day_range`
    filter and the `estimated_bbox_area_cm2` percentile mode both need a
    per-species, per-season MIN/MAX(datetime) baseline that has to be
    computed in Python (the `datetime` column mixes EXIF-style and ISO
    timestamp formats, so it can't be reliably compared/aggregated in raw
    SQL) before they can be turned into a filter at all.

    Per-species filter order (fixed, not configurable):
      1. category_common_name + season + year -- in SQL. This is the ONLY
         thing that defines the candidate pool used to compute the
         day_range baseline below -- is_primary, extends_border,
         blur_effect, non_target_weed, non_target_weed_pred_conf,
         num_components, and estimated_bbox_area_cm2 must never skew where
         that window sits, since none of them are date-defining. (Before
         this was fixed, is_primary/extends_border were applied in this
         same SQL query, which could anchor the day_range window to an
         artificially sparse subset -- see project conversation for the
         concrete case that surfaced this.)
      2. day_range window, if set -- computed in Python from the step-1
         rows' own MIN/MAX(datetime).
      3. is_primary / extends_border, applied in Python to the step-2 rows.
         Each defaults to the global `morphological.is_primary` /
         `extends_border` value, but a species may override it: blank
         (unset) inherits the global value, True/False overrides it for
         that species only, and the literal string "any" explicitly opts
         that species out of the filter entirely (even while it's still
         active for every other species). blur_effect/non_target_weed/
         non_target_weed_pred_conf remain purely global (no per-species
         override), applied here too.
      4. num_components range, if set.
      5. estimated_bbox_area_cm2, if set -- either an absolute {min,max}
         range, or a percentile threshold computed over the step-4 rows
         for that season.

    The sqlite connection is opened strictly read-only -- this class never
    writes to the database.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.db_path = cfg.paths.sql_database
        self.table_name = cfg.sqlite3.cutouts
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.species_filters = cfg.cutout_filters.category.species_filters
        self.morphological = cfg.cutout_filters.morphological

    def close(self) -> None:
        self.conn.close()

    # ---- global (uniform by default, per-species overridable) morphological filters ----
    # Applied in Python, AFTER the day_range window (see class docstring for
    # why) -- never folded into the SQL fetch that defines the day_range
    # baseline.

    @staticmethod
    def _resolve_species_override(species_value: Any, global_value: Any) -> Optional[bool]:
        """Resolve is_primary/extends_border for one species:
          - blank (None)      -> inherit the global morphological value
          - True / False      -> override for this species only
          - the literal string "any" (case-insensitive) -> explicit opt-out:
            no filter for this species, even if the global filter is active
            for everyone else.
        """
        if species_value is None:
            return global_value
        if isinstance(species_value, str) and species_value.strip().lower() == "any":
            return None
        return species_value

    def _apply_global_morphological_filters(
        self, rows: List[sqlite3.Row], is_primary: Optional[bool], extends_border: Optional[bool]
    ) -> List[sqlite3.Row]:
        m = self.morphological
        non_target_weed = m.get("non_target_weed")
        blur = m.get("blur_effect") or {}
        blur_min, blur_max = blur.get("min"), blur.get("max")
        ntw_conf = m.get("non_target_weed_pred_conf") or {}
        ntw_min, ntw_max = ntw_conf.get("min"), ntw_conf.get("max")

        if m.get("validated") is not None:
            log.warning(
                "morphological.validated is set (%r) but this database copy has no "
                "`validated` column -- ignoring this filter.", m.get("validated")
            )

        if all(v is None for v in (is_primary, extends_border, non_target_weed, blur_min, blur_max, ntw_min, ntw_max)):
            return rows

        def keep(row: sqlite3.Row) -> bool:
            if is_primary is not None and row["is_primary"] != is_primary:
                return False
            if extends_border is not None and row["extends_border"] != extends_border:
                return False
            if non_target_weed is not None and row["non_target_weed"] != non_target_weed:
                return False
            if blur_min is not None and (row["blur_effect"] is None or row["blur_effect"] < blur_min):
                return False
            if blur_max is not None and (row["blur_effect"] is None or row["blur_effect"] > blur_max):
                return False
            if ntw_min is not None and (row["non_target_weed_pred_conf"] is None or row["non_target_weed_pred_conf"] < ntw_min):
                return False
            if ntw_max is not None and (row["non_target_weed_pred_conf"] is None or row["non_target_weed_pred_conf"] > ntw_max):
                return False
            return True

        return [row for row in rows if keep(row)]

    # ---- per-species helpers ----

    def _seasons_for_species(self, common_name: str) -> List[str]:
        rows = self.conn.execute(
            f"SELECT DISTINCT season FROM {self.table_name} "
            "WHERE LOWER(TRIM(category_common_name)) = ?",
            (common_name.lower().strip(),),
        ).fetchall()
        return [r["season"] for r in rows]

    def _fetch_season_rows(self, common_name: str, season: str, year_filter) -> List[sqlite3.Row]:
        """Fetch the candidate pool for one species+season: species+season+year
        ONLY. Nothing else (not is_primary/extends_border, not
        num_components, not bbox) may narrow this pool, since it's what the
        day_range baseline gets computed from."""
        conditions = ["LOWER(TRIM(category_common_name)) = ?", "season = ?"]
        params: List[Any] = [common_name.lower().strip(), season]

        if year_filter:
            years = [str(y) for y in year_filter]
            placeholders = ", ".join("?" for _ in years)
            # substr(datetime, 1, 4) reliably extracts the year from both
            # datetime formats present in this db (EXIF and ISO both start
            # with a 4-digit year).
            conditions.append(f"substr(datetime, 1, 4) IN ({placeholders})")
            params.extend(years)

        query = f"SELECT * FROM {self.table_name} WHERE " + " AND ".join(conditions)
        return self.conn.execute(query, params).fetchall()

    @staticmethod
    def _day_range_mode(day_range) -> Tuple[Optional[str], Any]:
        """Validate day_range config and return (mode, value), or (None, None)
        if left entirely blank (no date filtering)."""
        if not day_range:
            return None, None

        last_days = day_range.get("last_days")
        first_days = day_range.get("first_days")
        trim_days = day_range.get("trim_days") or {}
        trim_start = trim_days.get("start")
        trim_end = trim_days.get("end")
        trim_active = trim_start is not None or trim_end is not None

        active_modes = [bool(x) for x in (last_days is not None, first_days is not None, trim_active)]
        if sum(active_modes) > 1:
            raise ValueError(
                "day_range must use exactly one of last_days / first_days / trim_days, got: "
                f"last_days={last_days!r}, first_days={first_days!r}, "
                f"trim_days={{'start': {trim_start!r}, 'end': {trim_end!r}}}"
            )

        if last_days is not None:
            return "last_days", last_days
        if first_days is not None:
            return "first_days", first_days
        if trim_active:
            return "trim_days", (trim_start or 0, trim_end or 0)
        return None, None

    @staticmethod
    def _apply_day_range(rows: List[sqlite3.Row], mode: str, value: Any) -> List[sqlite3.Row]:
        parsed = [(parse_datetime(row["datetime"]), row) for row in rows]
        parsed = [(dt, row) for dt, row in parsed if dt is not None]
        if not parsed:
            return []

        dts = [dt for dt, _ in parsed]
        min_dt, max_dt = min(dts), max(dts)

        if mode == "last_days":
            window_start, window_end = max_dt - timedelta(days=value), max_dt
        elif mode == "first_days":
            window_start, window_end = min_dt, min_dt + timedelta(days=value)
        else:  # trim_days
            start_days, end_days = value
            window_start = min_dt + timedelta(days=start_days)
            window_end = max_dt - timedelta(days=end_days)

        return [row for dt, row in parsed if window_start <= dt <= window_end]

    @staticmethod
    def _bbox_mode(bbox_cfg) -> Tuple[Optional[str], Any]:
        """Validate estimated_bbox_area_cm2 config and return (mode, value),
        or (None, None) if left entirely blank (no bbox filtering)."""
        if not bbox_cfg:
            return None, None

        min_v = bbox_cfg.get("min")
        max_v = bbox_cfg.get("max")
        pct = bbox_cfg.get("percentile")

        minmax_active = min_v is not None or max_v is not None
        pct_active = pct is not None

        if minmax_active and pct_active:
            raise ValueError(
                "estimated_bbox_area_cm2 must use exactly one of {min,max} or percentile, "
                f"got min={min_v!r}, max={max_v!r}, percentile={pct!r}"
            )
        if minmax_active:
            return "minmax", (min_v, max_v)
        if pct_active:
            return "percentile", pct
        return None, None

    @staticmethod
    def _apply_bbox_filter(rows: List[sqlite3.Row], mode: Optional[str], value: Any) -> List[sqlite3.Row]:
        if mode is None or not rows:
            return rows

        if mode == "minmax":
            min_v, max_v = value

            def within_range(row: sqlite3.Row) -> bool:
                area = row["estimated_bbox_area_cm2"]
                if area is None:
                    return False
                if min_v is not None and area < min_v:
                    return False
                if max_v is not None and area > max_v:
                    return False
                return True

            return [row for row in rows if within_range(row)]

        # percentile
        areas = sorted(
            row["estimated_bbox_area_cm2"] for row in rows if row["estimated_bbox_area_cm2"] is not None
        )
        if not areas:
            return []
        threshold = percentile(areas, value)
        return [
            row for row in rows
            if row["estimated_bbox_area_cm2"] is not None and row["estimated_bbox_area_cm2"] > threshold
        ]

    @staticmethod
    def _apply_num_components_filter(rows: List[sqlite3.Row], num_components_cfg) -> List[sqlite3.Row]:
        if not num_components_cfg:
            return rows
        min_v = num_components_cfg.get("min")
        max_v = num_components_cfg.get("max")
        if min_v is None and max_v is None:
            return rows

        def within_range(row: sqlite3.Row) -> bool:
            n = row["num_components"]
            if n is None:
                return False
            if min_v is not None and n < min_v:
                return False
            if max_v is not None and n > max_v:
                return False
            return True

        return [row for row in rows if within_range(row)]

    # ---- main entry point ----

    def fetch_all(self) -> List[Dict[str, Any]]:
        """Fetch filtered rows for every species in species_filters, applying
        each species' own season/year/day_range/num_components/bbox
        filters, and return the combined pool of rows as plain dicts."""
        all_rows: List[Dict[str, Any]] = []

        for common_name, spec in self.species_filters.items():
            spec = spec or {}
            season_filter = spec.get("season")
            year_filter = spec.get("year")
            seasons = list(season_filter) if season_filter else self._seasons_for_species(common_name)

            day_mode, day_value = self._day_range_mode(spec.get("day_range"))
            bbox_mode, bbox_value = self._bbox_mode(spec.get("estimated_bbox_area_cm2"))
            num_components_cfg = spec.get("num_components")
            is_primary = self._resolve_species_override(spec.get("is_primary"), self.morphological.get("is_primary"))
            extends_border = self._resolve_species_override(
                spec.get("extends_border"), self.morphological.get("extends_border")
            )

            species_kept = 0
            for season in seasons:
                rows = self._fetch_season_rows(common_name, season, year_filter)
                fetched = len(rows)

                if day_mode:
                    rows = self._apply_day_range(rows, day_mode, day_value)
                after_day_range = len(rows)

                rows = self._apply_global_morphological_filters(rows, is_primary, extends_border)
                after_morphological = len(rows)

                rows = self._apply_num_components_filter(rows, num_components_cfg)
                after_num_components = len(rows)

                rows = self._apply_bbox_filter(rows, bbox_mode, bbox_value)

                log.info(
                    "%s / %s: fetched=%d -> day_range=%d -> morphological=%d -> num_components=%d -> bbox=%d",
                    common_name, season, fetched, after_day_range, after_morphological, after_num_components, len(rows),
                )

                species_kept += len(rows)
                all_rows.extend(dict(row) for row in rows)

            log.info("Species '%s': %d rows kept across %d season(s).", common_name, species_kept, len(seasons))

        return all_rows


def main(cfg: DictConfig) -> List[Dict[str, Any]]:
    engine = SpeciesFilterQueryEngine(cfg)
    try:
        rows = engine.fetch_all()
        log.info("Fetched %d total rows across all species.", len(rows))
        return rows
    finally:
        engine.close()
