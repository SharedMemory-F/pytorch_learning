# -*- coding: utf-8 -*-
"""
author  : zhangyifan
date    : 2020-3-24
brief   : 逻辑回归模型得训练二分类问题
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np 
torch.manual_seed(10)
# step1 数据处理
# step2 模型选择
# step3 损失函数选择
# step4 优化器选择
# step5 模型训练
#------------step1 生成数据----------------------------
def create_datas():
    sample_nums = 100
    mean_value = 1.7
    bias = 1
    n_data = torch.ones(sample_nums, 2)# shape:(100, 2)
    # 以设置正态分布中不同的mean生成两堆数据
    # 类别0：均值mean_value, 标准差：1, 特征shape:(100, 2), 标签：0
    x0 = torch.normal(mean_value * n_data, 1) + bias
    y0 = torch.zeros(sample_nums)
    # 类别1：均值mean_value, 标准差：1，特征shape:(100, 2), 标签：1
    x1 = torch.normal(-mean_value * n_data, 1) + bias
    y1 = torch.ones(sample_nums)
    train_x = torch.cat((x0, x1), 0)
    train_y = torch.cat((y0, y1), 0)
    return train_x, train_y

class LogisticReg(nn.Module):
    # step2 选择模型
    def __init__(self):
        super(LogisticReg, self).__init__()
        self.features = nn.Linear(2, 1) #data shape:(100,2), weights shape:(2,1)
        self.sigmoid = nn.Sigmoid()
    # step3 选择损失函数
    def loss_fcn(self):
        return nn.BCELoss()
    # step4 选择优化器
    def optimizer(self, lr=0.01):
        return lr, torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
    # step5 开始训练
    def train(self):
        train_x, train_y = create_datas() # 加载数据
        loss_fcn = self.loss_fcn() # 加载loss
        lr, optimizer = self.optimizer() # 加载 lr，优化器
        def forward(train_x):# 前向计算
            return self.sigmoid(self.features(train_x))
        for i in range(1000):
            y_pred = forward(train_x) #前向计算
            loss = loss_fcn(y_pred.squeeze(), train_y) #loss计算
            loss.backward() #反向传播
            optimizer.step() #更新参数
            optimizer.zero_grad() #清空梯度
            # 训练信息显示
            if i % 10 == 0:
                mask = y_pred.ge(0.5).float().squeeze() #以0.5为阈值进行分类
                correct = (mask == train_y).sum() #计算正确预测的样本个数
                acc = correct.item() / train_y.size(0) #计算分类准确率
                # 将两类真实点绘出
                # plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c="r", label="class 0")
                # plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c="b", label="class 1")
                w0, w1 = self.features.weight[0]
                w0, w1 = float(w0.item()), float(w1.item())
                plot_b = float(self.features.bias.item())
                # plot_x = np.arange(-6, 6, 0.1)
                # plot_y = (-w0 * plot_x - plot_b) / w1
                # plt.xlim(-5, 7)
                # plt.ylim(-7, 7)
                # plt.plot(plot_x, plot_y)#绘出预测点
                # plt.text(-5, 5, 'loss: {:.4f}'.format(loss.data.numpy()))
                # plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(i, w0, w1, plot_b, acc))
                # plt.legend()
                # plt.show()
                print("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%} loss:{:.2f}".format(i, w0, w1, plot_b, acc, loss.data))
                plt.pause(0.5)
                if acc > 0.99:
                    break

if __name__ == "__main__":
    create_datas()
    lr = LogisticReg()
    lr.train()



