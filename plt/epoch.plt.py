import re
import matplotlib.pyplot as plt
import os

folder = '../runs/SIN/train_v5_4'
train_file = os.path.join(folder, 'metrics-train.txt')
val_file = os.path.join(folder, 'metrics-val.txt')
def read_metrics_file(filename):
    """从文件中读取指标字典，key是指标名，value是对应的列表"""
    metrics = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # 匹配形如 var_name=[...]  的格式
            m = re.match(r'(\w+)\s*=\s*\[(.*?)\]', line)
            if m:
                var_name = m.group(1)
                values_str = m.group(2)
                # 分割字符串，转换成float列表
                values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
                # 去掉train_epochs_或val_epochs_前缀，统一命名
                if var_name.startswith('train_epochs_'):
                    key = var_name[len('train_epochs_'):]
                elif var_name.startswith('val_epochs_'):
                    key = var_name[len('val_epochs_'):]
                else:
                    key = var_name
                metrics[key] = values
    return metrics

# 读取两个文件
train_data = read_metrics_file(train_file)
val_data = read_metrics_file(val_file)

# 取所有指标的合集（有些可能只有train或val有）
all_metrics = train_data.keys()

cols = 2
rows = (len(all_metrics) + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
axes = axes.flatten()

for i, metric in enumerate(all_metrics):
    ax = axes[i]
    # X轴epoch
    train_values = train_data.get(metric, [])
    val_values = val_data.get(metric, [])
    epochs_train = range(1, len(train_values) + 1)
    epochs_val = range(1, len(val_values) + 1)
    if train_values:
        ax.plot(epochs_train, train_values, label='Train')
    if val_values:
        ax.plot(epochs_val, val_values, label='Val')
    ax.set_title(metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

# 隐藏多余的子图
for j in range(len(all_metrics), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
