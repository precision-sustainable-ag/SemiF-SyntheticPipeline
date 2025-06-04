import os
import sys
import json
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf

# util imports
from utils.pdf import generate_pdf
from utils.utils import read_recipe
from utils.utils import resolve_image_storage_locations
from utils.graphs import scatter_plot, bar_chart_plot, jitter_plot, pie_chart

class CutoutAnalyzer():
    def __init__(self, query_type, param, states, cfg):

        self.cfg = cfg
        self.db_path = str(f"{cfg.paths.datadir}/db/agir.db")

        # Initialize dictionaries for stats
        self.batch_num_components = {}
        self.batch_image_dict = {}
        self.bbox = {}
        self.blur = {}
        self.rgb_mean_red = {}
        self.rgb_mean_green = {}
        self.rgb_mean_blue = {}
        self.rgb_std_red = {}
        self.rgb_std_green = {}
        self.rgb_std_blue = {}

        # KEEP TRACK OF STATES FOR COLOR COORDINATION BETWEEN GRAPHS
        self.states = states

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(semif_cutouts);")
        columns = [col[1] for col in cursor.fetchall()]

        # Find out which storage we would be getting the cutouts from
        batch_ids, cutout_ids = read_recipe(f"{cfg.paths.recipesdir}/{cfg.project_name}_{cfg.sub_name}.json")
        self.storage_location_data = resolve_image_storage_locations(batch_ids, cutout_ids, cfg)
        self.cutout_ids = cutout_ids

        if query_type == 'specified_configs': # load downloaded cutout metadata
            self.load_cutout_metadata(param, cursor, columns)
            self.graph_cutout_data("Downloaded Cutouts")
        elif query_type == 'specified_species': # load all data of species specified in config
            self.load_species_metadata(param, cursor, columns)
            self.graph_cutout_data("All Cutouts")

        conn.close()

    def load_cutout_metadata(self, images_dir, cursor, columns):

        # Loop through specified cutouts
        for cutout_id in self.cutout_ids:
            cursor.execute("SELECT * FROM semif_cutouts WHERE cutout_id = ?", (cutout_id,))
            rows = cursor.fetchall()
            self.query_for_metadata(rows, columns)

    def load_species_metadata(self, common_name, cursor, columns):

        # Loop through all specified species cutouts
        for species in common_name:
            species_lower = species.lower()
            cursor.execute(
                "SELECT * FROM semif_cutouts WHERE LOWER(json_extract(category, '$.common_name')) = ?",
                (species_lower,)
            )
            rows = cursor.fetchall()
            self.query_for_metadata(rows, columns)


    def query_for_metadata(self, rows, columns):
        for row in rows:
            row_dict = dict(zip(columns, row))
            try:
                row_dict['cutout_props'] = json.loads(row_dict['cutout_props'])
                row_dict['category'] = json.loads(row_dict['category'])
            except Exception:
                continue

            species = row_dict['category']['common_name'].upper()

            # Initialize species
            for d in [self.batch_image_dict, self.batch_num_components, self.bbox, self.blur,
                        self.rgb_mean_red, self.rgb_mean_green, self.rgb_mean_blue,
                        self.rgb_std_red, self.rgb_std_green, self.rgb_std_blue]:
                if species not in d:
                    d[species] = {}

            self.metadata_to_dict(species, row_dict['cutout_id'], row_dict)
            if self.states is None:
                self.states.append(row_dict['cutout_id'][:2])
            elif row_dict['cutout_id'][:2] not in self.states:
                self.states.append(row_dict['cutout_id'][:2])
                print(row_dict['cutout_id'][:2])

    def metadata_to_dict(self, species, synthetic, row_dict):
        self.append_image_size(species, synthetic, row_dict)
        self.append_num_components(species, synthetic, row_dict)
        self.append_bbox(species, synthetic, row_dict)
        self.append_blur(species, synthetic, row_dict)
        self.append_rgb_mean(species, synthetic, row_dict)
        self.append_rgb_std(species, synthetic, row_dict)

    def append_image_size(self, species, synthetic, cutout):
        self.batch_image_dict[species].setdefault(synthetic, []).append((cutout['cutout_height'], cutout['cutout_width']))

    def append_num_components(self, species, synthetic, cutout):
        self.batch_num_components[species].setdefault(synthetic, []).append(cutout['cutout_props']['num_components'])

    def append_bbox(self, species, synthetic, cutout):
        self.bbox[species].setdefault(synthetic, []).append(cutout['cutout_props']['bbox_area_cm2'])

    def append_blur(self, species, synthetic, cutout):
        self.blur[species].setdefault(synthetic, []).append(cutout['cutout_props']['blur_effect'])

    def append_rgb_mean(self, species, synthetic, cutout):
        r, g, b = cutout['cutout_props'].get('cropout_rgb_mean', [None, None, None])
        self.rgb_mean_red[species].setdefault(synthetic, []).append(r)
        self.rgb_mean_green[species].setdefault(synthetic, []).append(g)
        self.rgb_mean_blue[species].setdefault(synthetic, []).append(b)

    def append_rgb_std(self, species, synthetic, cutout):
        r, g, b = cutout['cutout_props'].get('cropout_rgb_std', [None, None, None])
        self.rgb_std_red[species].setdefault(synthetic, []).append(r)
        self.rgb_std_green[species].setdefault(synthetic, []).append(g)
        self.rgb_std_blue[species].setdefault(synthetic, []).append(b)

    def graph_cutout_data(self, title_info):
        
        file_path = self.cfg.paths.analysisdir

        title_info = title_info.replace(" ", "_").lower()
        os.makedirs(str(f"{file_path}/{title_info}"), exist_ok=True)

        # Grab graph colors
        cmap = plt.get_cmap("tab20")
        norm = mcolors.Normalize(vmin=0, vmax=len(self.states) - 1)
        palette = {state: mcolors.to_hex(cmap(norm(i))) for i, state in enumerate(self.states)}

        # Plot graphs
        for species in self.batch_image_dict:
            scatter_plot(self.batch_image_dict[species], species, "Height V Width", file_path, title_info, palette)
        for species in self.batch_num_components:
            bar_chart_plot(self.batch_num_components[species], species, "Number of Components", file_path, title_info, palette)
        for species in self.bbox:
            jitter_plot(self.bbox[species], species, "BBOX Area (cm^2)", file_path, title_info, palette)
        for species in self.blur:
            jitter_plot(self.blur[species], species, "Blur Effect", file_path, title_info, palette)

        # TODO EVAL USEFULNESS OF RGB GRAPHS
        # for species in self.rgb_mean_red:
        #     jitter_plot(self.rgb_mean_red[species], species, "mean_red")
        #     jitter_plot(self.rgb_mean_green[species], species, "mean_green")
        #     jitter_plot(self.rgb_mean_blue[species], species, "mean_blue")
        #     jitter_plot(self.rgb_std_red[species], species, "std_red")
        #     jitter_plot(self.rgb_std_green[species], species, "std_green")
        #     jitter_plot(self.rgb_std_blue[species], species, "std_blue")

        pie_chart(self.storage_location_data, "Cutout Distribution Across Storages", file_path)

def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)

    # Graph all species specified in config
    all_cutouts = CutoutAnalyzer("specified_species", cfg.cutout_filters.category.common_name, [], cfg)

    # Graph cutouts from local folder
    CutoutAnalyzer("specified_configs", cfg.paths.cutout_dir, all_cutouts.states, cfg)

    generate_pdf()