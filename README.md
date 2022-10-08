# SemiF-SyntheticPipeline


# Workflow

1. Choose cutouts by using configuration file
2. Generate synthetic data



# Data Structure

* Pot and background images are located in this repository
* Outputs (`images`, `masks`, and `metadata`) are also located in this repository.


## Pipeline
![](asset/Pipeline_flowchart.png)

<br>

## UML Class diagram

![](asset/Class_diagrams-synthetic.png)


```
SemiF-SyntheticPipeline
└── data
    └── semifield-synth
        ├── backgrounds
        │   ├── background_1.json
        │   └── background_1.png
        ├── pots
        │   ├── pot_1.json
        │   └── pot_1.png
        ├── images
        │   ├── c2cd770ad1ba4543803bbce4cc4b6184.png
        │   └── cccd3d7264624a72a3b11888bf2edde6.png
        ├── masks
        │   ├── c2cd770ad1ba4543803bbce4cc4b6184.png
        │   └── cccd3d7264624a72a3b11888bf2edde6.png
        └── metadata
            ├── c2cd770ad1ba4543803bbce4cc4b6184.json
            └── cccd3d7264624a72a3b11888bf2edde6.json
```  
* Cutouts are taken from the Semifield-cutout directory, in this example, another repository.

```
SemiF-AnnotationPipeline
└── data
    └── semifield-cutouts
        └── MD_2022-06-28
            ├── MD_2_2_1655826744.0_0.png
            └── MD_2_2_1655826744.0_0.json
```
<br>

## Data

### background_1.json
```Json
{
    "background_path": "backgrounds/background_1.png",
    "background_id": "4440b6a6-38da-463b-8d40-dde1b7b79900"
}
```

<br>

### pot_1.json
```Json
{
    "pot_path": "pots/MD_1_16_1656440273.0.png",
    "pot_id": "db9225b1-342f-483c-b350-85014b7bf611"
}
```

<br>

### Cutout example
```Json
{
    "blob_home": "data",
    "data_root": "semifield-cutouts",
    "batch_id": "MD_2022-06-21",
    "image_id": "MD_2_2_1655826744.0",
    "cutout_id": "MD_2_2_1655826744.0_0",
    "cutout_path": "MD_2022-06-21/MD_2_2_1655826744.0_0.png",
    "cls": "plant",
    "cutout_num": 0,
    "is_primary": false,
    "datetime": "2022:06:21 23:56:59",
    "cutout_props": {
        "area": 6117.0,
        "area_bbox": 17358.0,
        "area_convex": 7551.0,
        "axis_major_length": 270.7782922363416,
        "axis_minor_length": 31.13171551152259,
        "centroid0": 124.0601602092529,
        "centroid1": 193.53997057381068,
        "eccentricity": 0.9933688209359223,
        "solidity": 0.8100913786253476,
        "perimeter": 642.3229432149742
    },
    "schema_version": "1.0"
}
```
## Synthetic image

```Json
{
    "data_root": "semifield-synth",
    "synth_path": "images/cccd3d7264624a72a3b11888bf2edde6.png",
    "synth_maskpath": "masks/cccd3d7264624a72a3b11888bf2edde6.png",
    "pots": [
        {
            "pot_path": "pots/MD_72_2_1655836004.0.png",
            "pot_id": "ad4676e7-9fcc-4280-80ed-ddd5688c6d10"
        },
        {
            "pot_path": "pots/pot_5.png",
            "pot_id": "a5166e35-bd08-40ac-a741-d2d11ca30361"
        }
    ],
    "background": [
        {
            "background_path": "backgrounds/MD_72_2_1655836004.0_background.png",
            "background_id": "4e0a75d4-f3ef-4920-b671-a9d94b38f27c"
        }
    ],
    "cutouts": [
        {
            
            "blob_home": "data",
            "data_root": "semifield-cutouts",
            "batch_id": "MD_2022-06-28",
            "image_id": "MD_2_14_1656451343.0",
            "cutout_num": 53,
            "datetime": "2022:06:29 05:19:25",
            "cutout_props": {
                "area": 35006.0,
                "area_bbox": 51170.0,
                "area_convex": 37856.0,
                "axis_major_length": 281.0165308653915,
                "axis_minor_length": 164.29763429928514,
                "centroid0": 554.6574587213621,
                "centroid1": 354.5368222590413,
                "eccentricity": 0.8112822486437252,
                "solidity": 0.9247147083685545,
                "perimeter": 849.2935059634514
            },
            "cutout_path": "MD_2022-06-28/MD_2_14_1656451343.0_53.png",
            "cutout_id": "MD_2_14_1656451343.0_53",
            "cls": "dicot",
            "is_primary": false,
            "schema_version": "1.0"
        },
        {
            "blob_home": "data",
            "data_root": "semifield-cutouts",
            "batch_id": "MD_2022-06-28",
            "image_id": "MD_3_4_1656430853.0",
            "cutout_num": 4,
            "datetime": "2022:06:28 23:37:56",
            "cutout_props": {
                "area": 63096.0,
                "area_bbox": 91780.0,
                "area_convex": 66771.0,
                "axis_major_length": 381.4839752304773,
                "axis_minor_length": 216.45776748452977,
                "centroid0": 776.4104380626347,
                "centroid1": 184.30910041841005,
                "eccentricity": 0.823435519768727,
                "solidity": 0.9449611358224379,
                "perimeter": 1087.259018078045
            },
            "cutout_path": "MD_2022-06-28/MD_3_4_1656430853.0_4.png",
            "cutout_id": "MD_3_4_1656430853.0_4",
            "cls": "dicot",
            "is_primary": false,
            "schema_version": "1.0"
        }
    ],
    "synth_id": "a6fd06ab-862f-4fb4-954e-7be34dc82c0c"
}
```