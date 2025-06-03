import os
import hydra
import json
import sqlite3
from from_root import from_root
from omegaconf import DictConfig, OmegaConf
from graphs import scatter_plot, bar_chart_plot, jitter_plot

class CutoutAnalyzer():
    def __init__(self, images_dir=None, common_name=None):
        self.db_path = str(from_root("data/db/agir.db"))

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

        if images_dir: # load downloaded cutout metadata
            self.load_cutout_metadata(images_dir)
        elif common_name: # load all data of species specified in config
            self.load_species_metadata(common_name)

        self.graph_cutout_data()

    def load_cutout_metadata(self, images_dir):
        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(semif_cutouts);")
        columns = [col[1] for col in cursor.fetchall()]

        # Loop through images
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(".png"):
                cutout_id = filename[:-4]  # strip .png

                cursor.execute("SELECT * FROM semif_cutouts WHERE cutout_id = ?", (cutout_id,))
                rows = cursor.fetchall()

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

                    synthetic = cutout_id  # use cutout_id as synthetic image ID

                    self.append_image_size(species, synthetic, row_dict)
                    self.append_num_components(species, synthetic, row_dict)
                    self.append_bbox(species, synthetic, row_dict)
                    self.append_blur(species, synthetic, row_dict)
                    self.append_rgb_mean(species, synthetic, row_dict)
                    self.append_rgb_std(species, synthetic, row_dict)

        conn.close()
    
    def load_species_metadata(self, common_name):
        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(semif_cutouts);")
        columns = [col[1] for col in cursor.fetchall()]

        # Loop through images
        for species in common_name:
            species_lower = species.lower()
            cursor.execute(
                "SELECT * FROM semif_cutouts WHERE LOWER(json_extract(category, '$.common_name')) = ?",
                (species_lower,)
            )
            rows = cursor.fetchall()

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

                synthetic = species_lower  # use cutout_id as synthetic image ID

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

    def graph_cutout_data(self):
        file_path = from_root('analyze_images/cutouts/')
        os.makedirs(file_path, exist_ok=True)
        # for species in self.batch_image_dict:
        #     scatter_plot(self.batch_image_dict[species], species)
        # for species in self.batch_num_components:
        #     bar_chart_plot(self.batch_num_components[species], species)
        for species in self.bbox:
            jitter_plot(self.bbox[species], species, "bbox_area_cm2", file_path)
        for species in self.blur:
            jitter_plot(self.blur[species], species, "blur_effect", file_path)

        # TODO EVAL USEFULNESS OF RGB GRAPHS
        # for species in self.rgb_mean_red:
        #     jitter_plot(self.rgb_mean_red[species], species, "mean_red")
        #     jitter_plot(self.rgb_mean_green[species], species, "mean_green")
        #     jitter_plot(self.rgb_mean_blue[species], species, "mean_blue")
        #     jitter_plot(self.rgb_std_red[species], species, "std_red")
        #     jitter_plot(self.rgb_std_green[species], species, "std_green")
        #     jitter_plot(self.rgb_std_blue[species], species, "std_blue")

@hydra.main(version_base="1.2", config_path=str(from_root("conf")), config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)

    # Graph cutouts from local folder
    CutoutAnalyzer(str(from_root("data/cutouts")), None)

    # Graph all species specified in config
    CutoutAnalyzer(None, cfg.cutout_filters.category.common_name)

if __name__ == "__main__":
    main()