# THIRD PARTIES
import cv2
import json
import sys, os
from pathlib import Path
from from_root import from_root

# OUR IMPORTS

# GRAPHS (NOT PART OF SEMIF-PIPELINE)
sys.path.append(str(from_root("analyze_images")))
from graphs import scatter_plot, bar_chart_plot, bbox_plot


'''
    The following class takes in recipes and a synthetics and aims to generate meaningful statistics based on the cutouts.

        PARAMS:
            DICTIONARY
                SYNTHETIC IMAGE: RECIPE

    The reason for dictionary pass in is to allow grouping of statistics for mutliple recipes. If you want to anaylze one image
    just simply pass in the image and its recipe.     

    # exclude image format .jpg ending
    EX: SyntheticAnalyzer({'synthetic': '.../golden-vetch/mixed/recipes/golden-vetch_mixed.json'})

'''
class SyntheticAnalyzer():
    def __init__(self, species_list):

        # TAKE USERS DICTIONARY         => SYNTHETIC IMAGE: RECIPE
        # REPLACE IT                    => SYNTHETIC IMAGE: RECIPE (PASRABLE JSON OBJ)
        # THIS IS DONE IN FUNCTION LOAD RECIPES
        self.synthetic_image___recipe_jsonobj_dict = {}

        # DICTIONARIES WE WILL PASS INTO GRAPHS.PY

        # THESE DICTIONARIES ARE BASED ON THE META DATA COLLECTED ON THE CUTOUT
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

        # SET UP THE GRAPH DICTIONARIES TO SUPPORT ANALYSIS OF MUTLIPLE SPECIES
        for species in species_list:
            self.batch_num_components[species.upper()] = {}
            self.batch_image_dict[species.upper()] = {}
            self.bbox[species.upper()] = {}
            self.blur[species.upper()] = {}
            self.rgb_mean_red[species.upper()] = {}
            self.rgb_mean_green[species.upper()] = {}
            self.rgb_mean_blue[species.upper()] = {}
            self.rgb_std_red[species.upper()] = {}
            self.rgb_std_green[species.upper()] = {}
            self.rgb_std_blue[species.upper()] = {}
            print(species)

        # START ANALYSIS, COLLECT META DATA
        self.load_recipes()
        self.load_metadata()

        # GRAPH
        self.graph_synthetic_image_data()

        for species in species_list: 
            print(f"Total cutouts analyzed for species {species}: {len(self.batch_image_dict[species.upper()]['all'])}")

    def load_recipes(self):
        with open(str(from_root("analyze_images/jsons/bad.json")), 'r') as file:
            data = json.load(file)
            self.synthetic_image___recipe_jsonobj_dict["all"] = data

    def load_metadata(self):
        for synthetic, cutout_list in self.synthetic_image___recipe_jsonobj_dict.items():
            for cutout in cutout_list:
                # These fields are JSON-encoded strings, so decode them first
                cutout['cutout_props'] = json.loads(cutout['cutout_props'])
                cutout['category'] = json.loads(cutout['category'])

                species = cutout['category']['common_name'].upper()

                self.get_image_sizes(species, synthetic, cutout)
                self.get_num_components(species, synthetic, cutout)
                self.get_bbox(species, synthetic, cutout)
                self.get_blur(species, synthetic, cutout)
                self.get_rgb_mean(species, synthetic, cutout)
                self.get_rgb_std(species, synthetic, cutout)


    def get_image_sizes(self, species, synthetic, cutout):
        if synthetic not in self.batch_image_dict[species]:
            self.batch_image_dict[species][synthetic] = []
        value = (cutout['cutout_height'], cutout['cutout_width'])
        # if value not in self.batch_image_dict[species][synthetic]:
        self.batch_image_dict[species][synthetic].append(value)

    def get_num_components(self, species, synthetic, cutout):
        if synthetic not in self.batch_num_components[species]:
            self.batch_num_components[species][synthetic] = []
        value = (cutout['cutout_props']['num_components'])
        # if value not in self.batch_num_components[species][synthetic]:
        self.batch_num_components[species][synthetic].append(value)

    def get_bbox(self, species, synthetic, cutout):
        if synthetic not in self.bbox[species]:
            self.bbox[species][synthetic] = []
        value = (cutout['cutout_props']['bbox_area_cm2'])
        # if value not in self.bbox[species][synthetic]:
        self.bbox[species][synthetic].append(value)

    def get_blur(self, species, synthetic, cutout):
        if synthetic not in self.blur[species]:
            self.blur[species][synthetic] = []
        value = (cutout['cutout_props']['blur_effect'])
        # if value not in self.blur[species][synthetic]:
        self.blur[species][synthetic].append(value)
    
    def get_rgb_mean(self, species, synthetic, cutout):
        if synthetic not in self.rgb_mean_red[species]:
            self.rgb_mean_red[species][synthetic] = []
            self.rgb_mean_green[species][synthetic] = []
            self.rgb_mean_blue[species][synthetic] = []

        rgb = cutout.get("cutout_props", {}).get("cropout_rgb_mean", [None, None, None])
        red, green, blue = rgb[0], rgb[1], rgb[2]

        if red is not None and red not in self.rgb_mean_red[species][synthetic]:
            self.rgb_mean_red[species][synthetic].append(red)
        if green is not None and green not in self.rgb_mean_green[species][synthetic]:
            self.rgb_mean_green[species][synthetic].append(green)
        if blue is not None and blue not in self.rgb_mean_blue[species][synthetic]:
            self.rgb_mean_blue[species][synthetic].append(blue)

    def get_rgb_std(self, species, synthetic, cutout):
        if synthetic not in self.rgb_std_red[species]:
            self.rgb_std_red[species][synthetic] = []
            self.rgb_std_green[species][synthetic] = []
            self.rgb_std_blue[species][synthetic] = []

        rgb = cutout.get("cutout_props", {}).get("cropout_rgb_std", [None, None, None])
        red, green, blue = rgb[0], rgb[1], rgb[2]

        if red is not None and red not in self.rgb_std_red[species][synthetic]:
            self.rgb_std_red[species][synthetic].append(red)
        if green is not None and green not in self.rgb_std_green[species][synthetic]:
            self.rgb_std_green[species][synthetic].append(green)
        if blue is not None and blue not in self.rgb_std_blue[species][synthetic]:
            self.rgb_std_blue[species][synthetic].append(blue)

    def graph_synthetic_image_data(self):
        os.makedirs(from_root('analyze_images/graphs/'), exist_ok=True)
        for species in self.batch_image_dict.keys():    
            scatter_plot(self.batch_image_dict[species], species)
        for species in self.batch_num_components.keys():    
            bar_chart_plot(self.batch_num_components[species], species)
        for species in self.bbox.keys():    
            bbox_plot(self.bbox[species], species, "bbox_area_cm2")   
        for species in self.blur.keys():
            bbox_plot(self.blur[species], species, "blur_effect")  
        
        for species in self.rgb_mean_red.keys():
            bbox_plot(self.rgb_mean_red[species], species, "mean_red")
        for species in self.rgb_mean_green.keys():
            bbox_plot(self.rgb_mean_green[species], species, "mean_green")
        for species in self.rgb_mean_blue.keys():
            bbox_plot(self.rgb_mean_blue[species], species, "mean_blue")       
        for species in self.rgb_std_red.keys():
            bbox_plot(self.rgb_std_red[species], species, "std_red")
        for species in self.rgb_std_green.keys():
            bbox_plot(self.rgb_std_green[species], species, "std_green")
        for species in self.rgb_std_blue.keys():
            bbox_plot(self.rgb_std_blue[species], species, "std_blue")

test = SyntheticAnalyzer(["Hairy Vetch", "Cereal Rye"]) 