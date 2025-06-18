import os
import json
import sqlite3
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


# util imports
from utils.pdf import PDFDrafter
from utils.utils import clear_directory, read_recipe, query_for_cutout_metadata
from utils.graphs import horizontal_bar_chart_plot, jitter_plot, vertical_bar_chart_plot, boolean_horizontal_bar_chart_plot

log = logging.getLogger(__name__)

class CutoutAnalyzer():
    def __init__(self, analysis_type: str, states: list[str], cfg: DictConfig) -> None:

        self.cfg = cfg
        self.db_path = str(cfg.paths.sql_database)

        # Extract list of common names
        species_list = list(set(name.lower() for name in cfg.cutout_filters.category.common_name))
        sorted_species = sorted(species_list, key=lambda s: s.lower())

        # KEEP TRACK OF NUMBER OF CUTOUTS
        self.num_cutouts = {}

        # Initialize dictionaries for stats
        self.batch_num_components = {}
        self.bbox = {}
        self.blur = {}
        self.is_primary = {}
        self.extends_border = {}
        for species in sorted_species:
            species = species.upper()
            self.batch_num_components[species] = {}
            self.bbox[species] = {}
            self.blur[species] = {}
            self.is_primary[species] = {}
            self.extends_border[species] = {}
            self.num_cutouts[species] = 0

        # KEEP TRACK OF STATES FOR COLOR COORDINATION BETWEEN GRAPHS
        self.states = states

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(semif_cutouts);")
        columns = [col[1] for col in cursor.fetchall()]

        if analysis_type == 'specified': 
            # Find out which storage we would be getting the cutouts from
            batch_ids, cutout_ids = read_recipe(f"{cfg.paths.recipesdir}/{cfg.project_name}_{cfg.sub_name}.json")
            storage_location_data = resolve_image_storage_locations(batch_ids, cutout_ids, sorted_species, cfg)
            # load downloaded cutout metadata
            self.load_cutout_metadata(cursor, columns, cutout_ids)
            self.graph_cutout_data(analysis_type, storage_location_data)
        elif analysis_type == 'all': 
            # load all data of species specified in config
            self.load_species_metadata(sorted_species, cursor, columns)
            self.graph_cutout_data(analysis_type, None)

        conn.close()

    def load_cutout_metadata(self, cursor: sqlite3.Cursor, columns: list[str], cutout_ids: list[str]) -> None:
        """
            load_cutout_metadata: Function used to query for all metadata from cutouts in the generated recipes
        """
        # Loop through specified cutouts
        for cutout_id in cutout_ids:
            cursor.execute("SELECT * FROM semif_cutouts WHERE cutout_id = ?", (cutout_id,))
            rows = cursor.fetchall()
            self.query_for_metadata(rows, columns)

    def load_species_metadata(self, common_name: list[str], cursor: sqlite3.Cursor, columns: list[str]) -> None:
        """
            load_species_metadata: Function used to query for all metadata for all cutouts in the common_name list
        """
        # Loop through all specified species cutouts
        for species in common_name:
            species_lower = species.lower()
            cursor.execute(
                "SELECT * FROM semif_cutouts WHERE LOWER(json_extract(category, '$.common_name')) = ?",
                (species_lower,)
            )
            rows = cursor.fetchall()
            self.query_for_metadata(rows, columns)

    def query_for_metadata(self, rows: list[tuple], columns: list[str]) -> None:
        """
            query_for_metadata: Function grabs all metadata for a given cutout_id
        """
        for row in rows:
            row_dict = dict(zip(columns, row))
            try:
                row_dict['cutout_props'] = json.loads(row_dict['cutout_props'])
                row_dict['category'] = json.loads(row_dict['category'])
            except Exception:
                continue

            species = row_dict['category']['common_name'].upper()

            self.metadata_to_dict(species, row_dict['cutout_id'], row_dict)
            if self.states is None:
                self.states.append(row_dict['cutout_id'][:2])
            elif row_dict['cutout_id'][:2] not in self.states:
                self.states.append(row_dict['cutout_id'][:2])
                # log states that we are pulling data from
                log.info(f"Cutouts pulled from: {row_dict['cutout_id'][:2]}")

    def metadata_to_dict(self, species: str, synthetic: str, cutout: dict) -> None:
        """
            metadata_to_dict: Function takes metadata and stores them in dictionaries to be passed in to graphing functions using Function graph_cutout_data
        """
        prop_map = {
            'num_components': self.batch_num_components,
            'bbox_area_cm2': self.bbox,
            'blur_effect': self.blur,
            'is_primary': self.is_primary,
            'extends_border': self.extends_border,
        }
        for prop, target_dict in prop_map.items():
            val = cutout['cutout_props'].get(prop)
            target_dict[species].setdefault(synthetic, []).append(val)

        # Track cutout count
        self.num_cutouts[species.upper()] += 1

    def graph_cutout_data(self, title_info: str, storage_location_data: dict[str, dict[str, int]] | None) -> None:
        """
            graph_cutout_data: Function takes metadata dictionaries and passes them into graphing functions
        """
        file_path = self.cfg.paths.analysisdir

        title_info = title_info.replace(" ", "_").lower()
        os.makedirs(str(f"{file_path}/{title_info}"), exist_ok=True)

        # Grab graph colors. Used to distinguish between states.
        colors = [
            "#d9a89e",  #  pinkish red
            "#aacc97",  #  mint green
            "#e9bc8f",  #  peach
            "#bfd1dd",  #  pastel blue
            "#c7c0d8",  #  lavender
            "#f7d794",  #  pastel apricot
            "#a8dadc",  #  pastel aqua
        ]

        palette = {state: colors[i % len(colors)] for i, state in enumerate(self.states)}

        for species in self.batch_num_components:
            logrithmic = (title_info == 'all')
            horizontal_bar_chart_plot(self.batch_num_components[species], species, "Number of Components", file_path, title_info, palette, logrithmic)
        for species in self.bbox:
            jitter_plot(self.bbox[species], species, "BBOX Area (cm^2)", file_path, title_info, palette)
        for species in self.blur:
            jitter_plot(self.blur[species], species, "Blur Effect", file_path, title_info, palette)
        for species in self.is_primary:
            boolean_horizontal_bar_chart_plot(self.is_primary[species], species, "Is Primary", file_path, title_info, palette)
        for species in self.extends_border:
            boolean_horizontal_bar_chart_plot(self.extends_border[species], species, "Extends Border", file_path, title_info, palette)

        os.makedirs(str(f"{file_path}/storage_location"), exist_ok=True)
        if storage_location_data:
            for species in storage_location_data:
                vertical_bar_chart_plot(storage_location_data[species], "Cutout Distribution Across Storages", file_path, 'storage_location', species)

def resolve_image_storage_locations(batch_ids: list[str], cutout_ids: list[str], all_species: list[str], cfg: DictConfig) -> dict[str, dict[str, int]]:

    data = {}

    primary_storage_base_downloads = secondary_storage_base_downloads = tertiary_storage_base_downloads = 0

    for species in all_species:
        species = species.upper()
        data[species] =  {
            f"primary: {cfg.paths.primary_longterm_storage}": 0,
            f"secondary: {cfg.paths.secondary_longterm_storage}": 0,
            f"tertiary: {cfg.paths.tertiary_longterm_storage}": 0
        }

    for batch_id, cutout_id in zip(batch_ids, cutout_ids):
        image_filename = f"{cutout_id}.png"

        species = query_for_cutout_metadata(cutout_id, cfg)

        # List of storage locations in order of preference.
        storages = [
            ("primary", Path(Path(cfg.paths.primary_longterm_storage, "semifield-cutouts"), batch_id, image_filename)),
            ("secondary", Path(Path(cfg.paths.secondary_longterm_storage, "semifield-cutouts"), batch_id, image_filename)),
            ("tertiary", Path(Path(cfg.paths.tertiary_longterm_storage, "semifield-cutouts"), batch_id, image_filename))
        ]

        # Try each storage location until the image is found and copied
        primary_storage_base_downloads = 0
        secondary_storage_base_downloads = 0
        tertiary_storage_base_downloads = 0
        for storage_name, storage_path in storages:
            if storage_path.exists():
                if storage_name == "primary":
                    primary_storage_base_downloads = 1
                elif storage_name == "secondary":
                    secondary_storage_base_downloads = 1
                elif storage_name == "tertiary":
                    tertiary_storage_base_downloads = 1
                break  # Exit .

        data[species][f"primary: {cfg.paths.primary_longterm_storage}"] += primary_storage_base_downloads
        data[species][f"secondary: {cfg.paths.secondary_longterm_storage}"] += secondary_storage_base_downloads
        data[species][f"tertiary: {cfg.paths.tertiary_longterm_storage}"] += tertiary_storage_base_downloads

    return data

def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)

    directory_for_graphs_of_all_cutouts = "all"
    directory_for_graphs_of_specified_cutouts = "specified"
    directory_for_graphs_of_storages = 'storage_location'

    if not cfg.cutout_filters.category.common_name:
        log.error("No species specified for analysis")
        log.error("Species can be specified in cutout_filters under common_name")
        return

    # Graph all species specified in config
    all_cutouts = CutoutAnalyzer(directory_for_graphs_of_all_cutouts, [], cfg)

    # Graph cutouts from local folder
    specified_cutouts = CutoutAnalyzer(directory_for_graphs_of_specified_cutouts, all_cutouts.states, cfg)

    # Total num of cutouts
    num_cutouts = all_cutouts.num_cutouts, specified_cutouts.num_cutouts

    # Create PDF from graphs
    report = PDFDrafter(cfg, num_cutouts)

    # Heading
    heading = "Analysis of Cutouts"

    # Description
    species_list = list(set(name.lower() for name in cfg.cutout_filters.category.common_name))
    sorted_species = sorted(species_list, key=lambda s: s.lower())
    if len(sorted_species) > 1:
        titled = [s.title() for s in sorted_species]
        species_str = ', '.join(titled[:-1]) + f", and {titled[-1]}"
    else:
        species_str = sorted_species[0].title()
    description = f"The following is a report of the {species_str} in the database. The aim is to display the metadata of all cutouts vs cutouts you specified in your configuration"

    report.initialize_heading_and_description(heading, description)
    report.add_graphs_to_pdf()
    report.save_pdf()

    # Delete graphs
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_all_cutouts}")
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_specified_cutouts}")
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_storages}")