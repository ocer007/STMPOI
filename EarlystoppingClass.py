import torch


class EarlystoppingClass:
    def __init__(self, patience=25, min_delta=0.001, path='checkpoint.pt', verbose=False, initial_epochs=50):
        """
        :param patience: 允许验证集损失不减少的epoch数
        :param min_delta: 损失减少的最小幅度
        :param path: 模型权重保存路径
        :param verbose: 是否输出日志信息
        :param initial_epochs: 前多少个epoch不进行早停判断
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.verbose = verbose
        self.epoch_count = 0  # 记录当前的epoch数
        self.initial_epochs = initial_epochs  # 记录不早停的初始epoch数

    def __call__(self, val_loss, model):
        self.epoch_count += 1  # 每次调用时增加epoch计数

        # 前initial_epochs个epoch不进行早停判断
        if self.epoch_count <= self.initial_epochs:
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前模型状态"""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased, saving model to {self.path}")
