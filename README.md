# STMPOI

## Introduction

STMPOI is a spatio-temporal and multimodal next POI recommendation model. It integrates user check-in trajectories, spatial-temporal dependency graphs, and multimodal POI information to improve next POI prediction.

## Requirements

The experiments are conducted under the following environment:

```text
torch==2.0.0
torch-summary==1.4.5
numpy==1.24.4
pandas==1.1.5
prettytable==2.0.0
matplotlib==3.3.4
scipy==1.6.1
tqdm==4.58.0
data==0.4
```

## Dataset

The processed check-in datasets are stored in the `dataset/` directory. The check-in data and preprocessing procedure follow GETNext:

```text
https://github.com/songyangco/GETNext
```

The original Foursquare check-in dataset is available at:

```text
https://sites.google.com/site/yangdingqi/home/foursquare-dataset
```

The multimodal POI information, including reviews, images, descriptions, categories, and locations, is collected from Foursquare:

```text
https://foursquare.com/city-guide
```

The processed PHO dataset used in our experiments is available at:

```text
https://drive.google.com/file/d/1wEKgQtsx_L6Biare_pHqskaSaiKntKYd/view?usp=sharing
```

After downloading, place the dataset under the `dataset/` directory, for example:

```text
dataset/
└── PHO/
    ├── PHO_train.csv
    ├── PHO_val.csv
    ├── PHO_test.csv
    └── ...
```

## Running Scripts

Before training, build the required graph files:

```bash
python build_graph.py
python build_graph_in_out.py
python build_graph_geohash.py
python build_geo_graph.py
python build_dtw_graph.py
python build_mm_graph.py
python build_ui_graph.py
```

Then run `train_ab_9.py`. For example, training on the PHO dataset:

```bash
python train_ab_9.py --data-train ../dataset/PHO/PHO_train.csv --data-val ../dataset/PHO/PHO_val.csv \
                --data-adj-mtx ../dataset/PHO/graph_A.csv --data-node-feats ../dataset/PHO/graph_X.csv \
                --data-adj-mtx-in ../dataset/PHO/in_graph_A.csv  --data-adj-mtx-out ../dataset/PHO/out_graph_A.csv \
                --data-UI-mtx ../dataset/PHO/graph_UI.csv \
                --data-geo-mtx ../dataset/PHO/graph_geo.csv --data-dtw-mtx ../dataset/PHO/graph_dtw.csv --data-mm-mtx ../dataset/PHO/graph_mm.csv \
                --geo-adj-mtx ../dataset/PHO/geo_graph_A.csv --geo-node-feats ../dataset/PHO/geo_graph_X.csv \
                --poi-image-embedding ../dataset/PHO/PHO_image_encoded.json \
                --poi-comment-embedding ../dataset/PHO/PHO_POI_comments_encoded.json \
                --poi-meta-embedding ../dataset/PHO/PHO_POI_meta_encoded_combined.json \
                --dataset-dir ../dataset/PHO \
                --device cuda:0 \
                --name train_ab_9 \
                --project runs/PHO \
                --batch 32
```
