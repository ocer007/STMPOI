import numpy as np
import os
import pandas as pd
import torch
from collections import defaultdict
from dtw import dtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 1: 读取POI访问数据并计算每个POI的24小时访问次数
def get_poi_hourly_visits_from_local_time(file_path):
    """
    从train.csv文件读取数据，
    直接从local_time字段提取小时（0~23），
    统计每个POI_id在24小时的访问次数。

    返回:
        dict: {POI_id: 长度24的list，表示每个小时的访问次数}
    """
    df = pd.read_csv(file_path)

    # 先把local_time转成datetime格式，方便提取小时
    df['local_time'] = pd.to_datetime(df['local_time'])

    # 提取小时 (0-23)
    hours = df['local_time'].dt.hour

    # 创建一个默认字典，value是长度24的np数组
    poi_hourly_counts = defaultdict(lambda: np.zeros(24, dtype=int))

    # 统计每个POI的每个小时访问次数
    for poi_id, hour in zip(df['POI_id'], hours):
        poi_hourly_counts[poi_id][hour] += 1

    # 转成普通dict，list格式
    poi_hourly_dict = {k: v.tolist() for k, v in poi_hourly_counts.items()}
    return poi_hourly_dict


def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def build_dtw_graph(poi_hourly_visits, poi_id2idx_dict):
    """
    接受一个POI的访问字典，并计算DTW距离图
    返回：归一化的DTW距离矩阵
    """
    # 根据poi_id2idx_dict排序POI，确保顺序一致
    sorted_poi_ids = sorted(poi_hourly_visits.keys(), key=lambda x: poi_id2idx_dict.get(x, -1))

    # 提取POI的时序数据，按照索引排序
    sorted_poi_hourly_visits = [poi_hourly_visits[poi_id] for poi_id in sorted_poi_ids]
    poi_time_series = np.array(sorted_poi_hourly_visits, dtype=np.float32)

    # 每个POI归一化访问量
    poi_time_series_norm = np.array([normalize(ts) for ts in poi_time_series])

    n = poi_time_series_norm.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    # 将数据转移到GPU
    poi_time_series_tensor = torch.tensor(poi_time_series_norm).to(device)

    # 计算DTW距离矩阵（对称矩阵，只计算上三角）
    for i in tqdm(range(n), desc="Calculating DTW", ncols=100):  # 添加进度条
        for j in range(i, n):
            # 转为GPU张量
            x = poi_time_series_tensor[i].unsqueeze(0)  # 变为1x24的张量
            y = poi_time_series_tensor[j].unsqueeze(0)  # 变为1x24的张量

            # DTW 计算
            dist, _, _, _ = dtw(x.cpu().detach().numpy(), y.cpu().detach().numpy(), dist=euclidean)

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 每行归一化距离向量到0-1
    dist_matrix_norm = np.zeros_like(dist_matrix)
    for i in range(n):
        dist_matrix_norm[i] = normalize(dist_matrix[i])
        dist_matrix_norm[i] = 1 - dist_matrix_norm[i]

    return dist_matrix_norm


def save_graph(adjacency_matrix, dst_dir, location):
    """
    保存DTW图到CSV文件
    """
    pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dst_dir, 'graph_dtw.csv'), index=False, header=False)
    print(f'{location} DTW graph processing completed.\n')


# 主程序
if __name__ == '__main__':
    locations = ['SIN', 'NYC', 'PHO']

    for location in locations:
        print(f'Processing {location} data...')

        # 设置路径
        src_dir = f'../dataset/{location}'
        file_path = os.path.join(src_dir, f'{location}_train.csv')

        # 获取POI的每小时访问次数
        poi_hourly_visits = get_poi_hourly_visits_from_local_time(file_path)

        # 加载 POI 到索引的映射
        data_node_feats = os.path.join(src_dir, f'graph_X.csv')
        nodes_df = pd.read_csv(data_node_feats)
        poi_id2idx_dict = {poi_id: idx for idx, poi_id in enumerate(nodes_df['node_name/poi_id'])}

        # 构建DTW图并保存
        adjacency_matrix = build_dtw_graph(poi_hourly_visits, poi_id2idx_dict)

        # 保存DTW图
        save_graph(adjacency_matrix, src_dir, location)
