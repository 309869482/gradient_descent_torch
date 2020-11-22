import torch

'''
batch_n = 1
# 通过隐藏层后输出的特征数
hidden_layer1 = 3
hidden_layer2 = 3
# 输入数据的特征个数
input_data = 2
# 最后输出的分类结果数
output_data = 2
'''

s = input("请输入学号：")

s0 = s[0:2]
s1 = s[2:4]
s2 = s[4:6]
s3 = s[6:8]

s0 = int(s0)
s1 = int(s1)
s2 = int(s2)
s3 = int(s3)

s0 = float(s0)
s1 = float(s1)
s2 = float(s2)
s3 = float(s3)

# 初始化权重
x = torch.FloatTensor([(s0+s3)/200, (s1+s3)/200])  # 1*2
x = x.reshape(-1, 2)
x = x.t()  # 2*1
print("x:{}\n".format(x))

y = torch.FloatTensor([(s2+s3)/200, s3/100])  # 1*2
y = y.reshape(-1, 2)
y = y.t()  # 2*1
print("y:{}\n".format(y))

w1 = torch.FloatTensor([[0.1, 0.8], [0.4, 0.6], [0.3, 0.5]])  # 3*2

w2 = torch.FloatTensor([[0.9, 0.2, 0.7], [0.6, 0.3, 0.7], [0.1, 0.8, 0.5]])  # 3*3

w3 = torch.FloatTensor([[0.4, 0.3, 0.5], [0.6, 0.2, 0.8]])  # 2*3
# 定义训练次数和学习效率
epoch_n = 1
learning_rate = 0.9

for epoch in range(epoch_n):
    h1 = torch.mm(w1, x)  # 3*1第一层隐藏层
    print("h1:{}\n".format(h1))
    y1 = torch.sigmoid(h1, out=None)  # 3*1 第一层输出
    print("y1:{}\n".format(y1))
    h2 = torch.mm(w2, y1)  # 3*1 第二层隐藏层
    print("h2:{}\n".format(h2))
    y2 = torch.sigmoid(h2, out=None)  # 3*1 第二层输出
    print("y2:{}\n".format(y2))
    h3 = torch.mm(w3, y2)  # 3*1
    print("h3:{}\n".format(h3))
    y3 = torch.sigmoid(h3, out=None)  # 2*1 输出
    print("y3:{}\n".format(y3))
    error = 0.5 * (y3 - y).pow(2)  # 误差
    loss = 0.5*(y3 - y).pow(2).sum()  # 损失函数
    print("Epoch:{}, Loss{:.4f}\n".format(epoch, loss))
    if loss <=0.001:
        break

    gradient_y3 = (y3 - y)  # 2*1 dL/dy3

    # 输出——隐藏层残差
    r3 = -gradient_y3*y3*(torch.IntTensor([[1], [1]])-y3)
    print("r3:{}\n".format(r3))

    # 输出——隐藏层加权求和
    a3 = torch.mm(w3.t(), r3)

    # 隐藏——隐藏层残差
    r2 = a3 * y2 * (torch.IntTensor([[1], [1], [1]]) - y2)
    print("r2:{}\n".format(r2))

    # 输出——隐藏层加权求和
    a2 = torch.mm(w2.t(), r2)

    # 隐藏——隐藏层残差
    r1 = a2 * y1 * (torch.IntTensor([[1], [1], [1]]) - y1)
    print("r1:{}\n".format(r1))

    # 更新w3
    delta_w3 = -0.9*torch.mm(r3, y2.t())
    print("delta_w3:{}\n".format(delta_w3))
    w3 = w3 - delta_w3
    print("w3:{}\n".format(w3))

    # 更新w2
    delta_w2 = -0.9 * torch.mm(r2, y1.t())
    print("delta_w2:{}\n".format(delta_w2))
    w2 = w2 - delta_w2
    print("w2:{}\n".format(w2))

    # 更新w1
    delta_w1 = -0.9 * torch.mm(r1, x.t())
    print("delta_w1:{}\n".format(delta_w1))
    w1 = w1 - delta_w1
    print("w1:{}\n".format(w1))
