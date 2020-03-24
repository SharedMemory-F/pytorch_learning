# -*- coding:utf-8 -*-
"""
@author : zhangyifan
@date   : 2020-3-24
@biref  : 一元线性回归模型
"""
import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)


class LinearReg:
    def __init__(self, lr=0.05, epoch=1000):
        self.lr = lr
        self.epoch = epoch
    
    # 创建训练数据
    def creat_train_datas(self):
        # x = torch.rand(20, 1)*10 #0-1均匀分布，size为（20，1）
        x = torch.randint(0, 10, size=(20,1))
        y = 2*x + (5 + torch.randn(20,1)) # randn标准正太分布，扰动bias=5
        return x, y
    
    def train(self, x, y):
        # 初始化线性回归参数
        w = torch.randn((1), requires_grad=True) # 一个数，需要记录梯度
        b = torch.zeros((1), requires_grad=True) 
        for i in range(self.epoch):
            # 前向传播
            wx = torch.mul(w, x)
            y_pred = torch.add(wx, b)

            # 计算 MSE 损失
            loss = (0.5 * (y - y_pred)**2).mean()

            # 反向传播获取梯度
            loss.backward()

            # 更新参数, 加_ in-place操作，直接修改w.data
            w.data.sub_(self.lr * w.grad)
            b.data.sub_(self.lr * b.grad)
            
            # 需放在更新后，w=w-lr*w.grad会认为是中间结点，而非叶子结点
            w.grad.zero_()
            b.grad.zero_()

            # 训练过程绘图
            if i % 10 == 0:
                plt.scatter(x.data.numpy(), y.data.numpy()) #transform np to draw
                plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
                plt.text(2, 20, 'Loss={}'.format(loss.data.numpy()))
                plt.xlim(0, 10)
                plt.ylim(8, 28)
                plt.title("Iteration: {}\nw: {} b: {}".format(i, w.data.numpy(), b.data.numpy()))
                plt.pause(0.5)
                print("Iteration: {}\nw: {} b: {} loss:{}".format(i, w.data.numpy(), b.data.numpy(), loss.data.numpy()))

                if loss.data.numpy() < 0.5:
                    break

        return w, b

if __name__ == "__main__":
    lr_test = LinearReg()
    x, y = lr_test.creat_train_datas()
    w, b = lr_test.train(x, y)

    
