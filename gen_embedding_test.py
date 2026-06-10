import json
import logging
import os
import pathlib
import pickle
import zipfile
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from EarlystoppingClass import EarlystoppingClass
from pushplus import send_pushplus_message
from dataloader import load_graph_adj_mtx, load_graph_node_features, TrajectoryDatasetTrain, TrajectoryDatasetVal
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel, \
    MultiTransformerModel, MultiEmbReshape, POIMoEv4
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, \
    input_traj_to_embeddings, adjust_pred_prob_by_graph, set_seed, load_multimodal_data, \
    compute_average_embedding_v4, convert_gen_embeddings


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='_')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_A_in = load_graph_adj_mtx(args.data_adj_mtx_in)
    raw_A_out = load_graph_adj_mtx(args.data_adj_mtx_out)
    raw_X = load_graph_node_features(args.data_node_feats, args.feature1, args.feature2, args.feature3, args.feature4)
    num_pois = raw_X.shape[0]

    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {len(list(one_hot_encoder.categories_[0]))}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    A_in = calculate_laplacian_matrix(raw_A_in, mat_type='hat_rw_normd_lap_mat')
    A_out = calculate_laplacian_matrix(raw_A_out, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_id2idx_dict = {poi_id: idx for idx, poi_id in enumerate(nodes_df['node_name/poi_id'])}

    # Cat id to index
    cat_id2idx_dict = {cat_id: idx for idx, cat_id in enumerate(set(nodes_df[args.feature2]))}

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # Print user-trajectories count
    traj_list = list(set(train_df['traj_id'].tolist()))
    print(len(traj_list))

    # load geo data
    geo_mtx = pd.read_csv(args.data_geo_mtx, header=None).to_numpy()

    POI_gen_embedding = load_multimodal_data(args.poi_gen_desp, embedding_dim=args.NLP_embedding_dim)
    POI_multi_embedding = convert_gen_embeddings(
        poi_id2idx_dict,
        nodes_df,
        POI_gen_embedding,
        device=args.device,
        embedding_dim=args.NLP_embedding_dim
    )
    POI_multi_embedding_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in POI_multi_embedding.items()}
    poi_multi_embeddings_list = list(POI_multi_embedding_tensor.values())  # 获取字典中所有的值
    poi_multi_embeddings_tensor = torch.stack(poi_multi_embeddings_list).to(device=args.device)

    print(1)


if __name__ == '__main__':
    from param_parser_MM3 import parameter_parser

    args = parameter_parser()
    set_seed(args.seed)

    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train(args)
    push_token = "9648f50f750046e2ad17437fa297b67a"
    send_pushplus_message(push_token, "train state", "train over")
