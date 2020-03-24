# -*- coding:utf-8 -*-
"""
@author : zhangyifan 
@date   : 2020-03-20
@brief  : 张量的创建
"""
import torch
import numpy as np 
#torch.manul_seed(1) # 保证随机初始化的参数相同

class CreateTensors:
    def __init__(self, create_method=1, combine_method=1, divide_method=1, select_method=1, change_method=1):
        self.m1 = create_method
        self.m2 = combine_method
        self.m3 = divide_method
        self.m4 = select_method
        self.m5 = change_method

    def create_tensor(self): 
        # 通过torch.tensor创建张量
        if self.m1 == 1:
            arr = np.ones((3,3))
            arr_to_tensor = torch.tensor(arr, device="cuda")# 将tensor放在GPU上
            print("arr:{},\ntensor:{}".format(arr,arr_to_tensor.data))
        
        # 通过torch.from_numpy创建张量, 该方式是浅拷贝,修改tensor也会修改np.arr
        if self.m1 == 2:
            arr = np.array([[1,2,3],[4,5,6]])
            arr_to_tensor = torch.from_numpy(arr)
            arr_to_tensor[0,0] = -1
            print("arr:{}\ntensor:{}".format(arr,arr_to_tensor.data))
        
        # 通过torch.zeros创建全0张量
        if self.m1 == 3:
            out_t = torch.tensor([])
            tensor = torch.zeros((3,3), out=out_t)# out参数也是浅拷贝，指向同一内存
            print("tensor:{}\nout_t:{}\n".format(tensor,out_t))
            print("the tensor's address:{}\nthe out_t's address:{}".format(id(tensor),id(out_t)))
        
        # 通过torch.full创建全为某数字的张量
        if self.m1 == 4:
            tensor = torch.full((3,3),1)
            print(tensor)

        # 通过torch.arange创建等差数列张量
        if self.m1 == 5:
            tensor = torch.arange(2, 10, 2) #参数3是步长
            print(tensor)
        
        # 通过torch.linspace创建均分数列张量
        if self.m1 == 6:
            tensor = torch.linspace(2, 10, 6) #参数3是数列长度
            print(tensor)
        
        # 通过torch.logspace从创建对数均分的数列张量
        if self.m1 == 7:
            tensor = torch.logspace(1,100)
            print(tensor)

        # 创建单位对角矩阵
        if self.m1 == 8:
            tensor = torch.eye(3)
            print(tensor)

        # 通过torch.normal创建正态分布张量,注意比较mean和std分别是张量/标量的四种组合
        if self.m1 == 9:
            mean = torch.arange(1, 5, dtype=torch.float)
            std = 1.0
            t_normal = torch.normal(mean, std)
            print(t_normal)
        
        # 生成0至n-1的随机排列
        if self.m1 == 10:
            torch.randperm(10)
    
    def combine_tensor(self):
        """演示张量的两种拼接方法"""
        tensor1 = torch.ones((2,3))
        # cat在已有维度上进行拼接
        if self.m2 == 1:
            t_0 = torch.cat([t, t], dim=0)
            t_1 = torch.cat([t, t, t], dim=1)
            print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))
        
        # stack在新创建的维度上拼接，如果是已有维度，则原有的后移，新建该维度
        if self.m2 == 2:
            t_0 = torch.stack([t,t],dim=0)
            print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))

    def divide_tensor(self):
        """演示张量的两种切分方法"""
        tensor1 = torch.ones((2,7))
        if self.m3 == 1:
            list_of_tensors = torch.chunk(tensor1, dim=1, chunks=3)
            for idx, t in enumerate(list_of_tensors):
                print("第{}个张量：{}，shape is {}".format(idx+1, t, t.shape))

        if self.m3 == 2:
            list_of_tensors = torch.split(tensor1, [3,3,1], dim=1)
            for idx, t in enumerate(list_of_tensors):
                print("第{}个张量：{}，shape is {}".format(idx+1, t, t.shape))

    def select_tensor(self):
        """演示索引张量的两种方法"""
        tensor1 = torch.randint(0, 9, size=(3,3))
        if m4 == 1:
            idx = torch.tensor([0, 2], dtype=torch.long)
            t_select = torch.index_select(t, dim=0, index=idx)
            print("t_selcct:{}".format(t_selcct))
        if m4 == 2:
            # mask模的方法只能返回一维向量
            mask = tensor1.le(5)#小于等于5为true
            t_select = torch.masked_select(tensor1, mask)
            print("t_select:{}".format(t_selcct))

    def change_tensor(self):
        tensor1 = torch.randperm(8)
        if self.m5 == 1:
            t_reshape = tensor.reshape(t,(-1,2,2)) #某维设置-1，该维由其他维度决定
            print("t_reshape:{}".format(t_reshape))
        
        if self.m5 == 2:
            # 交换两个维度
             t_reshape = tensor.reshape(t,(-1,2,2))
             t_transpose = torch.transpose(t_reshape, dim0=1, dim1=2)
             pritn("t_transpose:{}".format(t_transpose))
        
        if self.m5 == 3:
            # 压缩维度
            t = torch.rand((1, 2, 3, 1))
            t_sq = torch.squeeze(t)
            t_0 = torch.squeeze(t, dim=0)
            t_1 = torch.squeeze(t, dim=1)
            print(t.shape)
            print(t_sq.shape)
            print(t_0.shape)
            print(t_1.shape)

if __name__ == "__main__":
    print("------------------------------\n\
1.torch.tensor\n2.torch.fromnumpy\n3.torch.zeros\n\
4.torch.full\n5.torch.arrange\n6.torch.linspace\n\
7.torch.logspace\n8.torch.eye\n9.torch.normal\n\
10.torch.randperm\n----------------------------")
    m1 = int(input("请选择创建张量的方法:"))
#     print("-------------------------------\n\
# 拼接：1.torch.cat\t\t 2.torch.stack\n\
# --------------------------------------\n\
# 切分：1.torch.chunk\t\t 2.torch.split\n\
# --------------------------------------\n\
# 索引：1.torch.index_select\t2.torch.masked_select\n\
# ---------------------------------------\n\
# 变换：1.reshape\t2.transpose\t3.squuze")
#     m2 = int(input("请选择拼接张量的方式："))
#     m3 = int(input("请选择切分张量的方式："))
#     m4 = int(input("请选择索引张量的方式："))
#     m5 = int(input("请选择变换张量的方式："))
    ct = CreateTensors(m1)
    ct.create_tensor() 
    # ct.combine_tensor() 
    # ct.divide_tensor()  
    # ct.index_tensor()
    # ct.change_tensor()

