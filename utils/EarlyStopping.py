import os
import sys

import torch
import numpy as np
from tqdm import tqdm

# 定义 EarlyStopping 类，用于训练中检测验证损失是否改善，从而提前停止训练
class EarlyStopping:
    """当验证损失在一定 patience 内未改善时，提前终止训练"""
    def __init__(self, save_path, patience=6, verbose=False, delta=0.):
        """
        参数说明：
            save_path : 模型保存文件夹
            patience (int): 在验证损失无改进时等待的 epoch 数
            verbose (bool): 是否输出详细信息
            delta (float): 要求最小改进阈值
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf   # 初始化最佳验证损失为无穷大
        self.delta = delta

    # 每个 epoch 后调用，判断当前模型是否改善，并保存模型
    def __call__(self, model, val_loss, current_epoch, save_every_model=True):
        # 调试时保存一个临时模型
        torch.save(model.state_dict(), 'debug_model.ckpt')

        if save_every_model:
            model_save_path = self.save_path + '/model' + str(current_epoch) + '.ckpt'
            torch.save(model.state_dict(), model_save_path)

        # 将验证损失取负作为分数（损失越低，分数越高）
        score = -val_loss

        if self.best_score is None:
            # 第一次运行时，记录当前分数和保存模型
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 如果当前分数没有足够提高，则累加计数器
            self.counter += 1
            print(f'\033[0;33mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m')
            if self.counter >= self.patience:
                # 达到耐心阈值后，触发 early stopping 标记
                self.early_stop = True
        else:
            # 如果出现改进，则保存模型并重置计数器
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    # 保存当前最优模型
    def save_checkpoint(self, val_loss, model):
        '''当验证损失下降时，保存模型参数'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network' + '.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
#EarlyStopping.py 用于监控验证集损失，当连续若干个 epoch 内没有改进时，停止训练。
# 它在保存最优模型和记录训练日志方面起到关键作用。