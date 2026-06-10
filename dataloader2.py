import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime


def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径 (单位：千米)
    R = 6371.0
    distance = R * c
    return distance


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def load_graph_node_features_of_geo(path, feature1='checkin_cnt', feature2='poi_catid_code'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path, delimiter="\t")
    rlt_df = df[[feature1, feature2]]
    X = rlt_df.to_numpy()

    return X


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, train_df, user_id2idx_dict, poi_id2idx_dict, args):
        self.df = train_df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        self.user_id2idx_dict = user_id2idx_dict
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args

        self.max_seq_len = 0  # 用于存储最长序列的长度

        for traj_id in tqdm(set(train_df['traj_id'].tolist())):
            traj_df = train_df[train_df['traj_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [self.poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[self.args.time_feature].to_list()

            latitudes = traj_df['latitude'].to_list()
            longitudes = traj_df['longitude'].to_list()
            distence_interval = [haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1]) for i in
                                 range(len(latitudes) - 1)]
            distence_interval.insert(0, 0.0)

            time_format = "%Y-%m-%d %H:%M:%S"
            local_time = [datetime.strptime(t, time_format) for t in traj_df['local_time'].to_list()]
            time_interval = [(local_time[i + 1] - local_time[i]).total_seconds() / 60 for i in
                             range(len(local_time) - 1)]
            time_interval.insert(0, 0.0)

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i], distence_interval[i], time_interval[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1], distence_interval[i + 1], time_interval[i + 1]))

            if len(input_seq) < self.args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

            self.max_seq_len = max(self.max_seq_len, len(input_seq))

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    def get_max_sequence_length(self):
        # 返回最长序列长度
        return self.max_seq_len


class TrajectoryDatasetVal(Dataset):
    def __init__(self, df, user_id2idx_dict, poi_id2idx_dict, args):
        self.df = df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        self.user_id2idx_dict = user_id2idx_dict
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args

        self.max_seq_len = 0  # 用于存储最长序列的长度

        for traj_id in tqdm(set(df['traj_id'].tolist())):
            user_id = traj_id.split('_')[0]

            if user_id not in self.user_id2idx_dict:
                continue

            traj_df = df[df['traj_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [self.poi_id2idx_dict[each] for each in poi_ids if each in self.poi_id2idx_dict]
            time_feature = traj_df[self.args.time_feature].to_list()
            latitudes = traj_df['latitude'].to_list()
            longitudes = traj_df['longitude'].to_list()
            distence_interval = [haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1]) for i in
                                range(len(latitudes) - 1)]
            distence_interval.insert(0, 0.0)

            time_format = "%Y-%m-%d %H:%M:%S"
            local_time = [datetime.strptime(t, time_format) for t in traj_df['local_time'].to_list()]
            time_interval = [(local_time[i + 1] - local_time[i]).total_seconds() / 60 for i in
                             range(len(local_time) - 1)]
            time_interval.insert(0, 0.0)

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i], distence_interval[i], time_interval[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1], distence_interval[i + 1], time_interval[i + 1]))

            if len(input_seq) < self.args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

            # 更新最大序列长度
            self.max_seq_len = max(self.max_seq_len, len(input_seq))

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    def get_max_sequence_length(self):
        # 返回最长序列长度
        return self.max_seq_len
