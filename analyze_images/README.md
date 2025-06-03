Cutouts are stored in the following directory on sunny  
```bash
/mnt/research-projects/s/screberg/
```  
They are located in three specific sub directories
```bash
longterm_images
```  
```bash
longterm_images2
```  
```bash
GROW_DATA
```  
In each of these subdirectories, the direcotry that holds the cutouts is semifield_cutouts. It contains many subdirectories that contain the image, mask, and the metadata stored in a json.  
ex.  
```bash
MD_2022-06-21
...
--MD_Row-1_1655826588_0_mask.png
--MD_Row-1_1655826588_0.jpg
--MD_Row-1_1655826588_0.json
...
```  

The python file aims to gather some high level data on the cutouts, specifically the ones specified in the config.  
Data stored in the database looks like this.

```sql
{
                    "season": "cool_season_covers_2023_2024",
                    "datetime": "2024:01:03 11:36:24",
                    "bbot_version": "v2.0",
                    "batch_id": "NC_2024-01-03",
                    "image_id": "NC_1704296968",
                    "cutout_id": "NC_1704296968_16",
                    "cutout_num": 16,
                    "cutout_height": 73,
                    "cutout_width": 72,
                    "lens_model": "FE 55mm F1.8 ZA",
                    "validated": 1,
                    "cutout_props": {
                        "is_primary": true,
                        "extends_border": false,
                        "bbox_area_cm2": 0.513435623437156,
                        "blur_effect": 0.2812995222439135,
                        "num_components": 2,
                        "cropout_rgb_mean": [
                            0.21103873817411284,
                            0.17665935476169156,
                            0.14158832482764797
                        ],
                        "cropout_rgb_std": [
                            0.12006048714244091,
                            0.12350068689420683,
                            0.08790835573153497
                        ],
                        "non_target_weed": false,
                        "non_target_weed_pred_conf": 0.9999775886535645,
                        "plant_pred": "target"
                    },
                    "category": {
                        "class_id": 35,
                        "USDA_symbol": "SECE",
                        "EPPO": "SECCE",
                        "group": "monocot",
                        "class": "Liliopsida",
                        "subclass": "Commelinidae",
                        "order": "Cyperales",
                        "family": "Poaceae",
                        "genus": "Secale",
                        "species": "Cereale",
                        "common_name": "Cereal Rye",
                        "authority": "Linnaeus",
                        "growth_habit": "graminoid",
                        "duration": "annual",
                        "category": "cool season cover crop",
                        "multi_species_USDA_symbol": null,
                        "link": "https://plants.usda.gov/home/plantProfile?symbol=SECE",
                        "note": null,
                        "hex": "#ba3a1c",
                        "rgb": [
                            186,
                            58,
                            28
                        ]
                    },
                    "cutout_version": 1.0,
                    "_id": "e5d9a636-f027-499d-8bfc-5b7178fd6743"
                }
```


