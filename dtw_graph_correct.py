import numpy as np
import os
import pandas as pd


def invert_values_in_csv(input_csv_path, output_csv_path):
    """
    读取原始CSV文件，将每个值变成 1 - 原来的值，并保存为新的CSV文件
    """
    # 读取原始 CSV 文件
    df = pd.read_csv(input_csv_path, header=None)

    # 对每个值应用 1 - 原来的值 操作
    inverted_df = 1 - df

    # 保存新的 CSV 文件
    inverted_df.to_csv(output_csv_path, index=False, header=False)
    print(f"Processed and saved the inverted values to {output_csv_path}")

# 主程序部分
if __name__ == '__main__':
    locations = ['NYC', 'PHO']

    for location in locations:
        print(f'Processing {location} data...')

        # 设置路径
        src_dir = f'../dataset/{location}'
        input_file_path = os.path.join(src_dir, 'graph_dtw.csv')  # 原始文件路径
        output_file_path = os.path.join(src_dir, 'graph_dtw.csv')  # 输出文件路径

        # 读取原文件并反转每个值
        invert_values_in_csv(input_file_path, output_file_path)
