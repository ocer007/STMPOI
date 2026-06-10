import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import load_multimodal_data, convert_gen_embeddings


def build_mm_graph(data_node_feats, poi_id2idx_dict, embedding_dim, device, embedding_file):

    # 读取 POI 数据
    nodes_df = pd.read_csv(data_node_feats)

    # 加载 POI 的多模态描述符（如 NLP 描述符）
    POI_gen_embedding = load_multimodal_data(embedding_file, embedding_dim=embedding_dim)

    # 使用 convert_gen_embeddings 函数获得 POI 的嵌入
    POI_multi_embedding = convert_gen_embeddings(
        poi_id2idx_dict,
        nodes_df,
        POI_gen_embedding,
        device=device,
        embedding_dim=embedding_dim
    )

    # 将嵌入转为 tensor
    POI_multi_embedding_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in POI_multi_embedding.items()}
    poi_multi_embeddings_list = list(POI_multi_embedding_tensor.values())
    poi_multi_embeddings = torch.stack(poi_multi_embeddings_list).to(device=device)

    # 计算余弦相似度矩阵
    cos_sim_matrix = cosine_similarity(poi_multi_embeddings.cpu().detach().numpy())

    # 返回相似度矩阵
    return cos_sim_matrix


def save_graph(adjacency_matrix, dst_dir, location):
    """
    Save the multimodal adjacency matrix to a CSV file.

    Args:
        adjacency_matrix (np.ndarray): Cosine similarity matrix to be saved.
        dst_dir (str): Directory where the CSV will be saved.
        location (str): Location name for file naming.
    """
    # 保存邻接矩阵到 CSV 文件
    pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dst_dir, f'graph_mm.csv'), index=False, header=False)
    print(f'{location} multimodal graph processing completed.\n')


if __name__ == '__main__':
    # 定义地点列表
    locations = ['SIN', 'NYC', 'PHO']

    # 遍历每个地点并处理
    for location in locations:
        print(f'Processing {location} data...')

        # 定义数据路径
        dst_dir = os.path.join('../dataset', location)

        data_node_feats = os.path.join(dst_dir, f'graph_X.csv')

        # 构建POI的 ID 到索引的映射
        nodes_df = pd.read_csv(data_node_feats)
        poi_id2idx_dict = {poi_id: idx for idx, poi_id in enumerate(nodes_df['node_name/poi_id'])}

        # 获取该地点的 POI embedding 文件路径
        embedding_file = os.path.join(dst_dir, f'{location}_gen_desp_embeddings.json')

        # 模拟模型配置，指定嵌入维度和设备
        embedding_dim = 768  # 你可以根据需要调整嵌入维度
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建 multimodal 邻接矩阵
        print(f'Building multimodal POI graph for {location} -----------------------------------')
        adjacency_matrix = build_mm_graph(data_node_feats, poi_id2idx_dict, embedding_dim, device, embedding_file)

        # 保存邻接矩阵到 CSV 文件
        save_graph(adjacency_matrix, dst_dir, location)

        print(f'{location} multimodal data processing completed.\n')
