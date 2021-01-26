#!/usr/bin/env python
# coding: utf-8

# ### 介绍

# 使用原生Pytorch实现GNN模型，在论文引用数据集Cora上进行训练和测试，实现的模型有论文[《The Graph Neural Network Model》](https://persagen.com/files/misc/scarselli2009graph.pdf)的Linear GNN，以及GCN模型(使用Pytorch实现PyG库的GCN模型)。

# ### 导入库

# In[1]:


import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ### 数据处理

# In[2]:


'''
node_num, feat_dim, stat_dim, num_class, T
feat_Matrix, X_Node, X_Neis, dg_list
'''
content_path = "./cora/cora.content"
cite_path = "./cora/cora.cites"

# 读取文本内容
with open(content_path, "r") as fp:
    contents = fp.readlines()
with open(cite_path, "r") as fp:
    cites = fp.readlines()

contents = np.array([np.array(l.strip().split("\t")) for l in contents])
paper_list, feat_list, label_list = np.split(contents, [1,-1], axis=1)
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)
# Paper -> Index dict
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])
# Label -> Index 字典
labels = list(set(label_list))
label_dict = dict([(key, val) for val, key in enumerate(labels)])
# Edge_index
cites = [i.strip().split("\t") for i in cites]
cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites], 
                 np.int64).T   # (2, edge)
cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
# Degree
_, degree_list = np.unique(cites[0,:], return_counts=True)

# Input
node_num = len(paper_list)
feat_dim = feat_list.shape[1]
stat_dim = 32
num_class = len(labels)
T = 2
feat_Matrix = torch.Tensor(feat_list.astype(np.float32))
X_Node, X_Neis = np.split(cites, 2, axis=0)
X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)),                  torch.from_numpy(np.squeeze(X_Neis))
dg_list = degree_list[X_Node]
label_list = np.array([label_dict[i] for i in label_list])
label_list = torch.from_numpy(label_list)


# ### 显示处理结果

# In[3]:


print("{}Data Process Info{}".format("*"*20, "*"*20))
print("==> Number of node : {}".format(node_num))
print("==> Number of edges : {}/2={}".format(cites.shape[1], int(cites.shape[1]/2)))
print("==> Number of classes : {}".format(num_class))
print("==> Dimension of node features : {}".format(feat_dim))
print("==> Dimension of node state : {}".format(stat_dim))
print("==> T : {}".format(T))
print("==> Shape of feat_Matrix : {}".format(feat_Matrix.shape))
print("==> Shape of X_Node : {}".format(X_Node.shape))
print("==> Shape of X_Neis : {}".format(X_Neis.shape))
print("==> Length of dg_list : {}".format(len(dg_list)))


# ### Linear GNN模型

# Linear GNN模型使用的是论文《The Graph Neural Network Model》中提到的Linear GNN模型，原论文将该模型使用在子图匹配问题上(本质上也是节点分类问题)，该实现将模型应用在Cora的数据集上。  
# 模型的观点大致如下：
# + 每个节点持有两个向量$\mathbf{x}_i$和$\mathbf{h}^t_i$，前者表示节点的特征向量，后者表示节点的状态向量，最初所有节点的初始化状态向量为0，即$\mathbf{h}^0_i=0,i=1,2...,N$。
# + 通过模型对节点的状态向量进行迭代更新，迭代次数为$T$次，直到达到不动点。
# + 使用$T$时刻(即不动点处)节点的状态向量和特征向量来得到节点的输出。

# In[4]:


'''
实现论文中的Xi函数，作为Hw函数的转换矩阵A，根据节点对(i,j)的特征向量
生成A矩阵，其中ln是特征向量维度，s为状态向量维度。
Initialization :
Input :
    ln : (int)特征向量维度
    s : (int)状态向量维度
Forward :
Input :
    x : (Tensor)节点对(i,j)的特征向量拼接起来，shape为(N, 2*ln)
Output :
    out : (Tensor)A矩阵，shape为(N, s, s)
'''
class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln   # 节点特征向量的维度
        self.s = s     # 节点的个数
        
        # 线性网络层
        self.linear = nn.Linear(in_features=2 * ln,
                                out_features=s ** 2,
                                bias=True)
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, X):
        bs = X.size()[0]
        out = self.linear(X)
        out = self.tanh(out)
        return out.view(bs, self.s, self.s)


'''
实现论文中的Rou函数，作为Hw函数的偏置项b
Initialization :
Input :
    ln : (int)特征向量维度
    s : (int)状态向量维度
Forward :
Input :
    x : (Tensor)节点的特征向量矩阵，shape(N, ln)
Output :
    out : (Tensor)偏置矩阵，shape(N, s)
'''
class Rou(nn.Module):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.linear = nn.Linear(in_features=ln,
                                out_features=s,
                                bias=True)
        self.tanh = nn.Tanh()
    def forward(self, X):
        return self.tanh(self.linear(X))

'''
实现Hw函数，即信息生成函数
Initialize :
Input :
    ln : (int)节点特征向量维度
    s : (int)节点状态向量维度
    mu : (int)设定的压缩映射的压缩系数
Forward :
Input :
    X : (Tensor)每一行为一条边的两个节点特征向量连接起来得到的向量，shape为(N, 2*ln)
    H : (Tensor)与X每行对应的source节点的状态向量
    dg_list : (list or Tensor)与X每行对应的source节点的度向量
Output :
    out : (Tensor)Hw函数的输出
'''
class Hw(nn.Module):
    def __init__(self, ln, s, mu=0.9):
        super(Hw, self).__init__()
        self.ln = ln
        self.s = s
        self.mu = mu
        
        # 初始化网络层
        self.Xi = Xi(ln, s)
        self.Rou = Rou(ln, s)
    
    def forward(self, X, H, dg_list):
        if isinstance(dg_list, list) or isinstance(dg_list, np.ndarray):
            dg_list = torch.Tensor(dg_list).to(X.device)
        elif isinstance(dg_list, torch.Tensor):
            pass
        else:
            raise TypeError("==> dg_list should be list or tensor, not {}".format(type(dg_list)))
        A = (self.Xi(X) * self.mu / self.s) / dg_list.view(-1, 1, 1)# (N, S, S)
        b = self.Rou(torch.chunk(X, chunks=2, dim=1)[0])# (N, S)
        out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)),-1) + b  # (N, s, s) * (N, s) + (N, s)
        return out    # (N, s)

'''
实现信息聚合函数，将前面使用Hw函数得到的信息按照每一个source节点进行聚合，
之后用于更新每一个节点的状态向量。
Initialize :
Input :
    node_num : (int)节点的数量
Forward :
Input :
    H : (Tensor)Hw的输出，shape为(N, s)
    X_node : (Tensor)H每一行对应source节点的索引，shape为(N, )
Output :
    out : (Tensor)求和式聚合之后的新的节点状态向量，shape为(V, s)，V为节点个数
'''
class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num
    
    def forward(self, H, X_node):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0,self.V-1).float(), 1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)

'''
实现Linear GNN模型，循环迭代计算T次，达到不动点之后，使用线性函数得到输出，进行
分类。
Initialize :
Input :
    node_num : (int)节点个数
    feat_dim : (int)节点特征向量维度
    stat_dim : (int)节点状态向量维度
    T : (int)迭代计算的次数
Forward :
Input :
    feat_Matrix : (Tensor)节点的特征矩阵，shape为(V, ln)
    X_Node : (Tensor)每条边的source节点对应的索引，shape为(N, )，比如`节点i->节点j`，source节点是`节点i`
    X_Neis : (Tensor)每条边的target节点对应的索引，shape为(N, )，比如`节点i->节点j`，target节点是`节点j`
    dg_list : (list or Tensor)与X_Node对应节点的度列表，shape为(N, )
Output :
    out : (Tensor)每个节点的类别概率，shape为(V, num_class)
'''
class OriLinearGNN(nn.Module):
    def __init__(self, node_num, feat_dim, stat_dim, num_class, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T
        # 输出层
        '''
        self.out_layer = nn.Sequential(
            nn.Linear(stat_dim, 16),   # ln+s -> hidden_layer
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(16, num_class)   # hidden_layer -> logits
        )
        '''
        self.out_layer = nn.Linear(stat_dim, num_class)
        self.dropout = nn.Dropout()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # 实现Fw
        self.Hw = Hw(feat_dim, stat_dim)
        # 实现H的分组求和
        self.Aggr = AggrSum(node_num)
        
    def forward(self, feat_Matrix, X_Node, X_Neis, dg_list):
        node_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Node)  # (N, ln)
        neis_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Neis)  # (N, ln)
        X = torch.cat((node_embeds, neis_embeds), 1)    # (N, 2 * ln)
        H = torch.zeros((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)  # (V, s)
        H = H.to(feat_Matrix.device)
        # 循环T次计算
        for t in range(self.T):
            # (V, s) -> (N, s)
            H = torch.index_select(H, 0, X_Neis)
            # (N, s) -> (N, s)
            H = self.Hw(X, H, dg_list)
            # (N, s) -> (V, s)
            H = self.Aggr(H, X_Node)
            # print(H[1])
        # out = torch.cat((feat_Matrix, H), 1)   # (V, ln+s)
        out = self.log_softmax(self.dropout(self.out_layer(H)))
        return out  # (V, num_class)


# ### 模型训练

# In[ ]:


# Split dataset
train_mask = torch.zeros(node_num, dtype=torch.uint8)
train_mask[:node_num - 1000] = 1                  # 1700左右training
val_mask = None                                    # 0valid
test_mask = torch.zeros(node_num, dtype=torch.uint8)
test_mask[node_num - 500:] = 1                    # 500test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OriLinearGNN(node_num, feat_dim, stat_dim, num_class, T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
feat_Matrix = feat_Matrix.to(device)
X_Node = X_Node.to(device)
X_Neis = X_Neis.to(device)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    # Get output
    out = model(feat_Matrix, X_Node, X_Neis, dg_list)
    
    # Get loss
    loss = F.nll_loss(out[train_mask], label_list[train_mask])
    _, pred = out.max(dim=1)
    
    # Get predictions and calculate training accuracy
    correct = float(pred[train_mask].eq(label_list[train_mask]).sum().item())
    acc = correct / train_mask.sum().item()
    print('[Epoch {}/200] Loss {:.4f}, train acc {:.4f}'.format(epoch, loss.cpu().detach().data.item(), acc))
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Evaluation on test data every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        _, pred = model(feat_Matrix, X_Node, X_Neis, dg_list).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))


# ### GCN模型

# GCN模型使用简单的两层GCN，实现方式使用原生Pytorch实现，模型和PyG的GCN模型的实现相同，只是框架使用Pytorch实现。  
# 关键要点如下：
# + 每个节点只保存一个特征向量$\mathbf{x}_i$，该特征向量的维度称为in_channel。
# + 每个节点向周围传递信息，对每个节点进行线性变换以得到该节点需要传递出去的信息。
# + 使用求和的方式来聚合信息。

# #### 准备数据(提前运行前面的“数据处理”部分)

# In[7]:


# Split dataset
train_mask = torch.zeros(node_num, dtype=torch.uint8)
train_mask[:node_num - 1000] = 1                  # 1700左右training
val_mask = None                                    # 0valid
test_mask = torch.zeros(node_num, dtype=torch.uint8)
test_mask[node_num - 500:] = 1                    # 500test
x = feat_Matrix
edge_index = torch.from_numpy(cites)


# In[75]:


'''
同Linear GNN
'''
class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num
    
    def forward(self, H, X_node):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0,self.V-1).float(), 1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)

'''
用于实现GCN的卷积块。
Initialize :
Input :
    in_channel : (int)输入的节点特征维度
    out_channel : (int)输出的节点特征维度
Forward :
Input :
    x : (Tensor)节点的特征矩阵，shape为(N, in_channel)，N为节点个数
    edge_index : (Tensor)边矩阵，shape为(2, E)，E为边个数。
Output :
    out : (Tensor)新的特征矩阵，shape为(N, out_channel)
'''
class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, node_num):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum(node_num)
        
    def forward(self, x, edge_index):
        # Add self-connect edges
        edge_index = self.addSelfConnect(edge_index, x.shape[0])
        
        # Apply linear transform
        x = self.linear(x)
        
        # Normalize message
        row, col = edge_index
        deg = self.calDegree(row, x.shape[0]).float()
        deg_sqrt = deg.pow(-0.5)  # (N, )
        norm = deg_sqrt[row] * deg_sqrt[col]
        
        # Node feature matrix
        tar_matrix = torch.index_select(x, dim=0, index=col)
        tar_matrix = norm.view(-1, 1) * tar_matrix  # (E, out_channel)
        # Aggregate information
        aggr =  self.aggregation(tar_matrix, row)  # (N, out_channel)
        return aggr
        
    
    def calDegree(self, edges, num_nodes):
        ind, deg = np.unique(edges.cpu().numpy(), return_counts=True)
        deg_tensor = torch.zeros((num_nodes, ), dtype=torch.long)
        deg_tensor[ind] = torch.from_numpy(deg)
        return deg_tensor.to(edges.device)
    
    def addSelfConnect(self, edge_index, num_nodes):
        selfconn = torch.stack([torch.range(0, num_nodes-1, dtype=torch.long)]*2,
                               dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index, selfconn],
                         dim=1)
    
        
'''
构建模型，使用两层GCN，第一层GCN使得节点特征矩阵
    (N, in_channel) -> (N, out_channel)
第二层GCN直接输出
    (N, out_channel) -> (N, num_class)
激活函数使用relu函数，网络最后对节点的各个类别score使用softmax归一化。
'''
class Net(nn.Module):
    def __init__(self, feat_dim, num_class, num_node):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16, num_node)
        self.conv2 = GCNConv(16, num_class, num_node)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


# In[ ]:


'''
开始训练模型
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feat_dim, num_class, node_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x = x.to(device)
edge_index = edge_index.to(device)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    # Get output
    out = model(x, edge_index)
    
    # Get loss
    loss = F.nll_loss(out[train_mask], label_list[train_mask])
    _, pred = out.max(dim=1)
    
    # Get predictions and calculate training accuracy
    correct = float(pred[train_mask].eq(label_list[train_mask]).sum().item())
    acc = correct / train_mask.sum().item()
    print('[Epoch {}/200] Loss {:.4f}, train acc {:.4f}'.format(epoch, loss.cpu().detach().data.item(), acc))
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Evaluation on test data every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        _, pred = model(x, edge_index).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))

