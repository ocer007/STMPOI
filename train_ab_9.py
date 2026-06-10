import json
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import zipfile
from collections import OrderedDict
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from EarlystoppingClass import EarlystoppingClass
from STtransformerv2 import TransformerModel1, TransformerModel2, TransformerModel3
from dataloader2 import load_graph_adj_mtx, load_graph_node_features, TrajectoryDatasetTrain, TrajectoryDatasetVal, \
    load_graph_node_features_of_geo
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel, \
    MultiTransformerModel, IntervalDistEncoder, STMoEv2
from pushplus import send_pushplus_message
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, \
    input_traj_to_embeddings, adjust_pred_prob_by_graph, set_seed, load_multimodal_data, \
    compute_average_embedding_v4, input_traj_to_embeddings_temp, input_traj_to_embeddings_temp5, \
    input_traj_to_embeddings_temp6, \
    convert_gen_embeddings, pad_matrix, expert_alignment_contrastive_loss


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='_')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
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
    # logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    # raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_A_in = load_graph_adj_mtx(args.data_adj_mtx_in)
    raw_A_out = load_graph_adj_mtx(args.data_adj_mtx_out)
    raw_X = load_graph_node_features(args.data_node_feats, args.feature1, args.feature2, args.feature3, args.feature4)
    num_pois = raw_X.shape[0]

    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")

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
    # A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
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
    print("Number of trajectories:", len(traj_list))

    # load geo data
    geo_mtx = pd.read_csv(args.data_geo_mtx, header=None).to_numpy()
    mm_mtx = pd.read_csv(args.data_mm_mtx, header=None).to_numpy()
    dtw_mtx = pd.read_csv(args.data_dtw_mtx, header=None).to_numpy()

    # Geo data load and one hot encode
    geo_raw_A = load_graph_adj_mtx(args.geo_adj_mtx)
    geo_raw_X = load_graph_node_features_of_geo(args.geo_node_feats, "checkin_cnt", "geohash")
    geo_list = list(geo_raw_X[:, 1])
    geo_one_hot_encoder = OneHotEncoder()
    geo_one_hot_encoder.fit(list(map(lambda x: [x], geo_list)))
    geo_one_hot_rlt = geo_one_hot_encoder.transform(list(map(lambda x: [x], geo_list))).toarray()
    with open(os.path.join(args.save_dir, 'geo-one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(geo_one_hot_encoder, f)

    A_geo = calculate_laplacian_matrix(geo_raw_A, mat_type='hat_rw_normd_lap_mat')
    geo_nodes_df = pd.read_csv(args.geo_node_feats, delimiter="\t")
    geo_ids = list(geo_nodes_df['geohash'].tolist())
    geo_id2idx_dict = OrderedDict(zip(geo_ids, range(len(geo_ids))))

    # Poi idx to geo idx
    poi_idx2geo_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2geo_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            geo_id2idx_dict[row['geohash']]

    num_geos = len(geo_ids)
    geo_embed_size = geo_one_hot_rlt.shape[-1]
    X_geo = geo_one_hot_rlt
    logging.info(f"After one hot encoding poi geo, X_geo.shape: {X_geo.shape}")

    # Multi data load
    save_path = os.path.join(args.dataset_dir, 'POI_multi_embedding_tensor.pkl')

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            POI_multi_embedding_tensor = pickle.load(f)
    else:
        POI_gen_embedding = load_multimodal_data(args.poi_gen_desp, embedding_dim=args.NLP_embedding_dim)
        POI_multi_embedding = convert_gen_embeddings(
            poi_id2idx_dict,
            nodes_df,
            POI_gen_embedding,
            device=args.device,
            embedding_dim=args.NLP_embedding_dim
        )
        POI_multi_embedding_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in POI_multi_embedding.items()}
        with open(save_path, "wb") as f:
            pickle.dump(POI_multi_embedding_tensor, f)
    poi_multi_embeddings_list = list(POI_multi_embedding_tensor.values())
    poi_multi_embeddings = torch.stack(poi_multi_embeddings_list).to(device=args.device)

    # user multi modal embedding
    UI_mtx = pd.read_csv(args.data_UI_mtx, header=None)
    UI_mtx.columns = range(UI_mtx.shape[1])
    zero_embedding = torch.zeros(args.NLP_embedding_dim)

    user_multi_embeddings = {}
    for user_id, row in tqdm(UI_mtx.iterrows(), total=len(UI_mtx)):
        poi_ids = row[row > 0].index
        interaction_values = row.loc[poi_ids].values
        embeddings = [
            POI_multi_embedding_tensor[poi_id] * count
            for poi_id, count in zip(poi_ids, interaction_values)
            if poi_id in POI_multi_embedding_tensor
        ]
        if embeddings:
            user_embedding = torch.stack(embeddings).mean(dim=0)
        else:
            user_embedding = zero_embedding.clone()
        user_multi_embeddings[user_id] = user_embedding

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df, user_id2idx_dict, poi_id2idx_dict, args)
    val_dataset = TrajectoryDatasetVal(val_df, user_id2idx_dict, poi_id2idx_dict, args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A_in = torch.from_numpy(A_in)
        A_out = torch.from_numpy(A_out)
    X = X.to(device=args.device, dtype=torch.float)
    A_in = A_in.to(device=args.device, dtype=torch.float)
    A_out = A_out.to(device=args.device, dtype=torch.float)

    if isinstance(X_geo, np.ndarray):
        X_geo = torch.from_numpy(X_geo)
        A_geo = torch.from_numpy(A_geo)
    X_geo = X_geo.to(device=args.device, dtype=torch.float)
    A_geo = A_geo.to(device=args.device, dtype=torch.float)

    # Define models
    args.gcn_nfeat = X.shape[1]
    poi_embed_model_in = GCN(ninput=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim,
                             dropout=args.gcn_dropout)
    poi_embed_model_out = GCN(ninput=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim,
                              dropout=args.gcn_dropout)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    user_embed_model = UserEmbeddings(len(user_id2idx_dict), args.user_embed_dim)
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
    geo_interval_embed_model = IntervalDistEncoder(interval_scale=3, interval_vocab_size=30,
                                                   hidden_size=args.time_embed_dim)
    time_interval_embed_model = IntervalDistEncoder(interval_scale=2, interval_vocab_size=30,
                                                    hidden_size=args.time_embed_dim)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    embed_fuse_model3 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

    args.gcn_geo_nfeat = X_geo.shape[1]
    geo_embed_model = GCN(ninput=args.gcn_geo_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.geo_embed_dim,
                          dropout=args.gcn_dropout)

    # Define sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model1 = TransformerModel1(num_pois, num_cats, args.seq_input_embed,
                                   args.transformer_nhead, args.transformer_nhid,
                                   args.transformer_nlayers, dropout=args.transformer_dropout)
    seq_model2 = TransformerModel2(num_pois, num_geos, args.seq_input_embed,
                                   args.transformer_nhead, args.transformer_nhid,
                                   args.transformer_nlayers, dropout=args.transformer_dropout)
    seq_model3 = TransformerModel3(num_pois, num_cats, 2 * args.NLP_embedding_dim,
                                   args.transformer_nhead, args.transformer_nhid,
                                   args.transformer_nlayers, dropout=args.transformer_dropout)

    st_moe_model = STMoEv2(num_pois, args.seq_input_embed, args.seq_input_embed, 2 * args.NLP_embedding_dim)

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model_in.parameters()) +
                                  list(poi_embed_model_out.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(geo_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(embed_fuse_model3.parameters()) +
                                  list(geo_interval_embed_model.parameters()) +
                                  list(time_interval_embed_model.parameters()) +
                                  list(seq_model1.parameters()) +
                                  list(seq_model2.parameters()) +
                                  list(seq_model3.parameters()) +
                                  list(st_moe_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_geo = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% ====================== Train ======================
    poi_embed_model_in = poi_embed_model_in.to(device=args.device)
    poi_embed_model_out = poi_embed_model_out.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    geo_embed_model = geo_embed_model.to(device=args.device)
    geo_interval_embed_model = geo_interval_embed_model.to(device=args.device)
    time_interval_embed_model = time_interval_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    embed_fuse_model3 = embed_fuse_model3.to(device=args.device)
    seq_model1 = seq_model1.to(device=args.device)
    seq_model2 = seq_model2.to(device=args.device)
    seq_model3 = seq_model3.to(device=args.device)
    st_moe_model = st_moe_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    train_epochs_align_loss_list = []

    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    val_epochs_align_loss_list = []

    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model_in.train()
        poi_embed_model_out.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        geo_embed_model.train()
        geo_interval_embed_model.train()
        time_interval_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        embed_fuse_model3.train()
        seq_model1.train()
        seq_model2.train()
        seq_model3.train()
        st_moe_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        train_batches_align_loss_list = []
        # Loop batch
        for b_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches")):
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_dtw_mtx = []
            batch_seq_geo_mtx = []
            batch_seq_mm_mtx = []
            batch_seq_embeds1 = []
            batch_seq_embeds2 = []
            batch_seq_embeds3 = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []

            poi_embeddings_in = poi_embed_model_in(X, A_in)
            poi_embeddings_out = poi_embed_model_out(X, A_out)
            poi_embeddings = poi_embeddings_in + poi_embeddings_out

            geo_embeddings = geo_embed_model(X_geo, A_geo)

            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                user_id = traj_id.split('_')[0]
                user_idx = user_id2idx_dict[user_id]
                input_seq = [each[0] for each in sample[1]]
                input_seq_time = [each[1] for each in sample[1]]
                input_seq_dtw_mtx = dtw_mtx[input_seq][:, input_seq]
                input_seq_geo_mtx = geo_mtx[input_seq][:, input_seq]
                input_seq_mm_mtx = mm_mtx[input_seq][:, input_seq]
                input_seq_embed_list1, input_seq_embed_list2, input_seq_embed_list3 = input_traj_to_embeddings_temp6(
                    sample, poi_embeddings, geo_embeddings, poi_multi_embeddings, user_multi_embeddings,
                    user_embed_model, time_embed_model, time_interval_embed_model, geo_interval_embed_model,
                    cat_embed_model,
                    embed_fuse_model1, embed_fuse_model2, embed_fuse_model3,
                    user_id2idx_dict, poi_idx2cat_idx_dict, poi_idx2geo_idx_dict,
                    device=args.device, args=args)
                input_seq_embed1 = torch.stack(input_seq_embed_list1)
                input_seq_embed2 = torch.stack(input_seq_embed_list2)
                input_seq_embed3 = torch.stack(input_seq_embed_list3)
                batch_seq_embeds1.append(input_seq_embed1)
                batch_seq_embeds2.append(input_seq_embed2)
                batch_seq_embeds3.append(input_seq_embed3)
                batch_seq_lens.append(len(input_seq))
                batch_seq_dtw_mtx.append(input_seq_dtw_mtx)
                batch_seq_geo_mtx.append(input_seq_geo_mtx)
                batch_seq_mm_mtx.append(input_seq_mm_mtx)
                batch_input_seqs.append(input_seq)

                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                lable_seq_geos = [poi_idx2geo_idx_dict[each] for each in label_seq]
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(lable_seq_geos))

            # Pad seqs for batch training
            batch_padded1 = pad_sequence(batch_seq_embeds1, batch_first=True, padding_value=-1)
            batch_padded2 = pad_sequence(batch_seq_embeds2, batch_first=True, padding_value=-1)
            batch_padded3 = pad_sequence(batch_seq_embeds3, batch_first=True, padding_value=-1)
            batch_dtw_padded = pad_matrix(batch_seq_dtw_mtx, padding_value=-1)
            batch_geo_padded = pad_matrix(batch_seq_geo_mtx, padding_value=-1)
            batch_mm_padded = pad_matrix(batch_seq_mm_mtx, padding_value=-1)

            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)

            src_mask = seq_model1.generate_square_subsequent_mask(batch_padded1.size(1)).to(args.device)
            if len(batch) != args.batch:
                src_mask = seq_model1.generate_square_subsequent_mask(batch_padded1.size(1)).to(args.device)

            # Feedforward
            x = batch_padded1.to(device=args.device, dtype=torch.float)
            x2 = batch_padded2.to(device=args.device, dtype=torch.float)
            x3 = batch_padded3.to(device=args.device, dtype=torch.float)
            x_dtw_mtx = batch_dtw_padded.to(device=args.device, dtype=torch.float)
            x_geo_mtx = batch_geo_padded.to(device=args.device, dtype=torch.float)
            x_mm_mtx = batch_mm_padded.to(device=args.device, dtype=torch.float)

            output1, y_pred_poi, y_pred_time = seq_model1(x, x_dtw_mtx, src_mask)
            output2, y_pred_poi2, y_pred_geo = seq_model2(x2, x_geo_mtx, src_mask)
            output3, y_pred_poi3, y_pred_cat = seq_model3(x3, x_mm_mtx, src_mask)

            out_poi1, out_poi2, out_poi3, y_pred_poi_sum = st_moe_model(output1, output2, output3)

            y_pred_time_sum = y_pred_time
            y_pred_geo_sum = y_pred_geo
            y_pred_cat_sum = y_pred_cat

            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            loss_poi = criterion_poi(y_pred_poi_sum.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time_sum), y_time)
            loss_geo = criterion_geo(y_pred_geo_sum.transpose(1, 2), y_geo)
            loss_cat = criterion_cat(y_pred_cat_sum.transpose(1, 2), y_cat)

            # Final loss
            align_loss = 0
            if "PHO" in args.data_adj_mtx_in or "NYC" in args.data_adj_mtx_in:
                align_loss = expert_alignment_contrastive_loss(out_poi1, out_poi2, out_poi3, y_pred_poi_sum)
                loss = loss_poi + loss_time * args.time_loss_weight + loss_geo + loss_cat + align_loss
            else:
                loss = loss_poi + loss_time * args.time_loss_weight + loss_geo + loss_cat

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_sum.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat_sum.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            train_batches_align_loss_list.append(align_loss.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                # logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                #              f'train_batch_loss:{loss.item():.2f}, '
                #              f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                #              f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                #              f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                #              f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                #              f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                #              f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                #              f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                #              f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                #              f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                #              f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                #              f'traj_id:{batch[sample_idx][0]}\n'
                #              f'input_seq: {batch[sample_idx][1]}\n'
                #              f'label_seq:{batch[sample_idx][2]}\n'
                #              f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                #              f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                #              f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                #              '=' * 100)
                # logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                #              f'train_batch_loss:{loss.item():.2f}, '
                #              f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                #              f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                #              f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                #              f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                #              f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                #              f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                #              f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                #              f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                #              f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                #              f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                #              )

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model_in.eval()
        poi_embed_model_out.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        geo_embed_model.eval()
        geo_interval_embed_model.eval()
        time_interval_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        embed_fuse_model3.eval()
        seq_model1.eval()
        seq_model2.eval()
        seq_model3.eval()
        st_moe_model.eval()

        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        val_batches_align_loss_list = []
        for vb_idx, batch in enumerate(tqdm(val_loader, desc="Valing Batches")):
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_dtw_mtx = []
            batch_seq_geo_mtx = []
            batch_seq_mm_mtx = []
            batch_seq_embeds1 = []
            batch_seq_embeds2 = []
            batch_seq_embeds3 = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []

            poi_embeddings_in = poi_embed_model_in(X, A_in)
            poi_embeddings_out = poi_embed_model_out(X, A_out)
            poi_embeddings = poi_embeddings_in + poi_embeddings_out

            geo_embeddings = geo_embed_model(X_geo, A_geo)

            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                user_id = traj_id.split('_')[0]
                user_idx = user_id2idx_dict[user_id]
                input_seq = [each[0] for each in sample[1]]
                input_seq_time = [each[1] for each in sample[1]]
                input_seq_dtw_mtx = dtw_mtx[input_seq][:, input_seq]
                input_seq_geo_mtx = geo_mtx[input_seq][:, input_seq]
                input_seq_mm_mtx = mm_mtx[input_seq][:, input_seq]
                input_seq_embed_list1, input_seq_embed_list2, input_seq_embed_list3 = input_traj_to_embeddings_temp6(
                    sample, poi_embeddings, geo_embeddings, poi_multi_embeddings, user_multi_embeddings,
                    user_embed_model, time_embed_model, time_interval_embed_model, geo_interval_embed_model,
                    cat_embed_model,
                    embed_fuse_model1, embed_fuse_model2, embed_fuse_model3,
                    user_id2idx_dict, poi_idx2cat_idx_dict, poi_idx2geo_idx_dict,
                    device=args.device, args=args)
                input_seq_embed1 = torch.stack(input_seq_embed_list1)
                input_seq_embed2 = torch.stack(input_seq_embed_list2)
                input_seq_embed3 = torch.stack(input_seq_embed_list3)
                batch_seq_embeds1.append(input_seq_embed1)
                batch_seq_embeds2.append(input_seq_embed2)
                batch_seq_embeds3.append(input_seq_embed3)
                batch_seq_lens.append(len(input_seq))
                batch_seq_dtw_mtx.append(input_seq_dtw_mtx)
                batch_seq_geo_mtx.append(input_seq_geo_mtx)
                batch_seq_mm_mtx.append(input_seq_mm_mtx)
                batch_input_seqs.append(input_seq)

                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                lable_seq_geos = [poi_idx2geo_idx_dict[each] for each in label_seq]
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(lable_seq_geos))

            # Pad seqs for batch training
            batch_padded1 = pad_sequence(batch_seq_embeds1, batch_first=True, padding_value=-1)
            batch_padded2 = pad_sequence(batch_seq_embeds2, batch_first=True, padding_value=-1)
            batch_padded3 = pad_sequence(batch_seq_embeds3, batch_first=True, padding_value=-1)
            batch_dtw_padded = pad_matrix(batch_seq_dtw_mtx, padding_value=-1)
            batch_geo_padded = pad_matrix(batch_seq_geo_mtx, padding_value=-1)
            batch_mm_padded = pad_matrix(batch_seq_mm_mtx, padding_value=-1)

            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)

            src_mask = seq_model1.generate_square_subsequent_mask(batch_padded1.size(1)).to(args.device)
            if len(batch) != args.batch:
                src_mask = seq_model1.generate_square_subsequent_mask(batch_padded1.size(1)).to(args.device)

            # Feedforward
            x = batch_padded1.to(device=args.device, dtype=torch.float)
            x2 = batch_padded2.to(device=args.device, dtype=torch.float)
            x3 = batch_padded3.to(device=args.device, dtype=torch.float)
            x_dtw_mtx = batch_dtw_padded.to(device=args.device, dtype=torch.float)
            x_geo_mtx = batch_geo_padded.to(device=args.device, dtype=torch.float)
            x_mm_mtx = batch_mm_padded.to(device=args.device, dtype=torch.float)

            output1, y_pred_poi, y_pred_time = seq_model1(x, x_dtw_mtx, src_mask)
            output2, y_pred_poi2, y_pred_geo = seq_model2(x2, x_geo_mtx, src_mask)
            output3, y_pred_poi3, y_pred_cat = seq_model3(x3, x_mm_mtx, src_mask)

            out_poi1, out_poi2, out_poi3, y_pred_poi_sum = st_moe_model(output1, output2, output3)
            y_pred_time_sum = y_pred_time
            y_pred_geo_sum = y_pred_geo
            y_pred_cat_sum = y_pred_cat

            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            loss_poi = criterion_poi(y_pred_poi_sum.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time_sum), y_time)
            loss_geo = criterion_geo(y_pred_geo_sum.transpose(1, 2), y_geo)
            loss_cat = criterion_cat(y_pred_cat_sum.transpose(1, 2), y_cat)

            # Final loss
            align_loss = 0
            if "PHO" in args.data_adj_mtx_in or "NYC" in args.data_adj_mtx_in:
                align_loss = expert_alignment_contrastive_loss(out_poi1, out_poi2, out_poi3, y_pred_poi_sum)
                loss = loss_poi + loss_time * args.time_loss_weight + loss_geo + loss_cat + align_loss
            else:
                loss = loss_poi + loss_time * args.time_loss_weight + loss_geo + loss_cat

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_sum.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat_sum.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            val_batches_align_loss_list.append(align_loss.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                # logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                #              f'val_batch_loss:{loss.item():.2f}, '
                #              f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                #              f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                #              f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                #              f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                #              f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                #              f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                #              f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                #              f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                #              f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                #              f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                #              f'traj_id:{batch[sample_idx][0]}\n'
                #              f'input_seq:{batch[sample_idx][1]}\n'
                #              f'label_seq:{batch[sample_idx][2]}\n'
                #              f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                #              f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                #              f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                #              f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                #              '=' * 100)
                # logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                #              f'val_batch_loss:{loss.item():.2f}, '
                #              f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                #              f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                #              f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                #              f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                #              f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                #              f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                #              f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                #              f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                #              f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                #              f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                #              )
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_train_align_loss = np.mean(train_batches_align_loss_list)

        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)
        epoch_val_align_loss = np.mean(val_batches_align_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_align_loss_list.append(epoch_train_align_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_align_loss_list.append(epoch_val_align_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(
            f"Epoch {epoch}/{args.epochs}\n"
            f"{'Train Losses:':<15} "
            f"{'total:':<8}{epoch_train_loss:<8.4f}"
            f"{'poi:':<8}{epoch_train_poi_loss:<8.4f}"
            f"{'time:':<8}{epoch_train_time_loss:<8.4f}"
            f"{'cat:':<8}{epoch_train_cat_loss:<8.4f}"
            f"{'align:':<8}{epoch_train_align_loss:<8.4f}\n"
            f"{'Train Metrics:':<15} "
            f"{'acc@1:':<8}{epoch_train_top1_acc:<8.4f}"
            f"{'acc@5:':<8}{epoch_train_top5_acc:<8.4f}"
            f"{'acc@10:':<8}{epoch_train_top10_acc:<8.4f}"
            f"{'acc@20:':<8}{epoch_train_top20_acc:<8.4f}"
            f"{'mAP20:':<8}{epoch_train_mAP20:<8.4f}"
            f"{'mrr:':<8}{epoch_train_mrr:<8.4f}\n"
            f"{'Val Losses:':<15} "
            f"{'total:':<8}{epoch_val_loss:<8.4f}"
            f"{'poi:':<8}{epoch_val_poi_loss:<8.4f}"
            f"{'time:':<8}{epoch_val_time_loss:<8.4f}"
            f"{'cat:':<8}{epoch_val_cat_loss:<8.4f}"
            f"{'align:':<8}{epoch_val_align_loss:<8.4f}\n"
            f"{'Val Metrics:':<15} "
            f"{'acc@1:':<8}{epoch_val_top1_acc:<8.4f}"
            f"{'acc@5:':<8}{epoch_val_top5_acc:<8.4f}"
            f"{'acc@10:':<8}{epoch_val_top10_acc:<8.4f}"
            f"{'acc@20:':<8}{epoch_val_top20_acc:<8.4f}"
            f"{'mAP20:':<8}{epoch_val_mAP20:<8.4f}"
            f"{'mrr:':<8}{epoch_val_mrr:<8.4f}\n"
        )

        # Save poi and user embeddings
        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, 'embeddings')
            if not os.path.exists(embeddings_save_dir): os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # # Save poi embeddings
                # poi_embeddings_in = poi_embed_model_in(X, A).detach().cpu().numpy()
                # poi_embeddings_out = poi_embed_model_out(X, A).detach().cpu().numpy()
                poi_embeddings = poi_embeddings_in + poi_embeddings_out
                poi_embedding_list = []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings)
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=args.device)
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten()
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings)
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=args.device)
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten()
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings)
                # Save time embeddings
                time_embedding_list = []
                for time_idx in range(args.time_units):
                    input = torch.FloatTensor([time_idx]).to(device=args.device)
                    time_embedding = time_embed_model(input).detach().cpu().numpy().flatten()
                    time_embedding_list.append(time_embedding)
                time_embeddings = np.array(time_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_in_state_dict': poi_embed_model_in.state_dict(),
                'poi_embed_out_state_dict': poi_embed_model_out.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                    print(epoch, file=f)
                max_val_score = monitor_score
        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_align_loss_list={[float(f"{each:.4f}") for each in train_epochs_align_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_align_loss_list={[float(f"{each:.4f}") for each in val_epochs_align_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)


if __name__ == '__main__':
    from param_parser_ab import parameter_parser

    args = parameter_parser()
    set_seed(args.seed)

    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train(args)
    push_token = "9648f50f750046e2ad17437fa297b67a"
    send_pushplus_message(push_token, "train state", "train_ab_1")
