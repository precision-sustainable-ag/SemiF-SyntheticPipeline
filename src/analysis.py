import os
import json
import sqlite3
import logging
from omegaconf import DictConfig, OmegaConf

# util imports
from utils.pdf import PDFDrafter
from utils.utils import clear_directory, read_recipe
from utils.graphs import bar_chart_plot, jitter_plot, barplot, boolean_bar_chart_plot

log = logging.getLogger(__name__)

class CutoutAnalyzer():
    def __init__(self, query_type, states, cfg) -> None:

        self.cfg = cfg
        self.db_path = str(cfg.paths.datadir)

        # Extract list of common names
        species_list = cfg.cutout_filters.category.common_name

        # Initialize dictionaries for stats
        self.batch_num_components = {}
        self.bbox = {}
        self.blur = {}
        self.is_primary = {}
        self.extends_border = {}
        self.rgb_mean_red = {}
        self.rgb_mean_green = {}
        self.rgb_mean_blue = {}
        self.rgb_std_red = {}
        self.rgb_std_green = {}
        self.rgb_std_blue = {}
        for species in species_list:
            species = species.upper()
            self.batch_num_components[species] = {}
            self.bbox[species] = {}
            self.blur[species] = {}
            self.is_primary[species] = {}
            self.extends_border[species] = {}
            self.rgb_mean_red[species] = {}
            self.rgb_mean_green[species] = {}
            self.rgb_mean_blue[species] = {}
            self.rgb_std_red[species] = {}
            self.rgb_std_green[species] = {}
            self.rgb_std_blue[species] = {}

        # KEEP TRACK OF STATES FOR COLOR COORDINATION BETWEEN GRAPHS
        self.states = states

        # KEEP TRACK OF NUMBER OF CUTOUTS
        self.num_cutouts = {}

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(semif_cutouts);")
        columns = [col[1] for col in cursor.fetchall()]

        if query_type == 'specified': 
            # Find out which storage we would be getting the cutouts from
            batch_ids, cutout_ids = read_recipe(f"{cfg.paths.recipesdir}/{cfg.project_name}_{cfg.sub_name}.json")
            storage_location_data = resolve_image_storage_locations(batch_ids, cutout_ids, cfg)
            # load downloaded cutout metadata
            self.load_cutout_metadata(cursor, columns, cutout_ids)
            self.graph_cutout_data(query_type, storage_location_data)
        elif query_type == 'all': 
            # load all data of species specified in config
            self.load_species_metadata(species_list, cursor, columns)
            self.graph_cutout_data(query_type, None)

        conn.close()

    def load_cutout_metadata(self, cursor, columns, cutout_ids) -> None:
        """
            load_cutout_metadata: Function used to query for all metadata from cutouts in the generated recipes
        """
        # Loop through specified cutouts
        for cutout_id in cutout_ids:
            cursor.execute("SELECT * FROM semif_cutouts WHERE cutout_id = ?", (cutout_id,))
            rows = cursor.fetchall()
            self.query_for_metadata(rows, columns)

    def load_species_metadata(self, common_name, cursor, columns) -> None:
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

    def query_for_metadata(self, rows, columns) -> None:
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

    def metadata_to_dict(self, species, synthetic, cutout) -> None:
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

        # NOTE: Commented out because not interested in it at this moment
        # # Handle RGB means
        # r, g, b = cutout['cutout_props'].get('cropout_rgb_mean', [None, None, None])
        # self.rgb_mean_red[species].setdefault(synthetic, []).append(r)
        # self.rgb_mean_green[species].setdefault(synthetic, []).append(g)
        # self.rgb_mean_blue[species].setdefault(synthetic, []).append(b)
        # # Handle RGB std
        # r, g, b = cutout['cutout_props'].get('cropout_rgb_std', [None, None, None])
        # self.rgb_std_red[species].setdefault(synthetic, []).append(r)
        # self.rgb_std_green[species].setdefault(synthetic, []).append(g)
        # self.rgb_std_blue[species].setdefault(synthetic, []).append(b)

        # Track cutout count
        self.num_cutouts.setdefault(species.upper(), 0)
        self.num_cutouts[species.upper()] += 1

    def graph_cutout_data(self, title_info, storage_location_data) -> None:
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
        ]
        palette = {state: colors[i % len(colors)] for i, state in enumerate(self.states)}

        for species in self.batch_num_components:
            logrithmic = (title_info == 'all')
            bar_chart_plot(self.batch_num_components[species], species, "Number of Components", file_path, title_info, palette, logrithmic)
        for species in self.bbox:
            jitter_plot(self.bbox[species], species, "BBOX Area (cm^2)", file_path, title_info, palette)
        for species in self.blur:
            jitter_plot(self.blur[species], species, "Blur Effect", file_path, title_info, palette)
        for species in self.is_primary:
            boolean_bar_chart_plot(self.is_primary[species], species, "Is Primary", file_path, title_info, palette)
        for species in self.extends_border:
            boolean_bar_chart_plot(self.extends_border[species], species, "Extends Border", file_path, title_info, palette)

        # TODO EVAL USEFULNESS OF RGB GRAPHS
        # for species in self.rgb_mean_red:
        #     jitter_plot(self.rgb_mean_red[species], species, "mean_red")
        #     jitter_plot(self.rgb_mean_green[species], species, "mean_green")
        #     jitter_plot(self.rgb_mean_blue[species], species, "mean_blue")
        #     jitter_plot(self.rgb_std_red[species], species, "std_red")
        #     jitter_plot(self.rgb_std_green[species], species, "std_green")
        #     jitter_plot(self.rgb_std_blue[species], species, "std_blue")

        os.makedirs(str(f"{file_path}/storage_location"), exist_ok=True)
        if storage_location_data:
            for species in storage_location_data:
                barplot(storage_location_data[species], "Cutout Distribution Across Storages", file_path, 'storage_location', species)

def resolve_image_storage_locations(batch_ids: list[str], cutout_ids: list[str], cfg) -> dict[str, dict[str, int]]:

    data = {}

    primary_storage_base_downloads = secondary_storage_base_downloads = tertiary_storage_base_downloads = 0
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

        if species not in data:
            data[species] =  {
                f"primary: {cfg.paths.primary_longterm_storage}": 0,
                f"secondary: {cfg.paths.secondary_longterm_storage}": 0,
                f"tertiary: {cfg.paths.tertiary_longterm_storage}": 0
            }

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
    PDFDrafter(cfg, num_cutouts)

    # Delete graphs
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_all_cutouts}")
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_specified_cutouts}")
    clear_directory(f"{cfg.paths.analysisdir}/{directory_for_graphs_of_storages}")