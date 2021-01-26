#!/usr/bin/env python
# coding: utf-8

# ## 导入库

# In[1]:


import os
import json
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ## 生成数据

# + **数据格式说明**  
#   数据集的格式为{graph1, graph2, ..., graphn}
#   其中，graphi为第i个graph数据，graphi的数据结构为G_i(N, E)：
#         N = {v_1, v_2, ..., v_n}  # n个节点集合
#         E = {e_1, e_2, ..., e_t}  # t条边集合
#    其中，e_j的数据结构为(v_p, v_q)，表示该边两端的节点。
# + **任务说明**  
#   给定子图sub_graph，训练集{graph1, graph2, ..., graphn}，模型目标就是对其中的graph中的节点进行分类。  
#   如果graphi包含sub_graph和节点v，且节点v属于sub_graph中的一个节点，那么节点v的标签为1，反之，标签为-1。模型就是要预测每个节点的标签是1还是-1。
# + **其它说明**  
#   这里使用的数据直接按照论文《The Graph Neural Network Model》所述的子图匹配任务进行实现，节点的特征就是数字1-10  
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
#   因此，节点集合使用节点特征的列表表示，边集合中的每一条边(i,j)中的i和j表示该边两端的节点在节点列表中对应的索引。

# In[2]:


'''
使用networkx库生成graph的图像，并显示，节点中显示的数字表示节点特征数字。
Input :
    graph : (Node_list, Edge_list)
'''
def graphFigure(graph):
    G = nx.Graph()
    node_feat_list, edge_list = graph
    node_names = ["v{}".format(i) for i in range(len(node_feat_list))]
    G.add_nodes_from(node_names, feat=node_feat_list)
    G.add_edges_from([("v{}".format(i),"v{}".format(j)) for i,j in edge_list])
    labels = dict([("v{}".format(i), feat) for i, feat in enumerate(node_feat_list)])
    plt.figure(0)
    nx.draw_spectral(G=G, 
                     node_size=1000,
                     node_color="g",
                     with_labels=True, 
                     font_weight='bold',
                     labels=labels)
'''
用于产生一个graph数据，首先随机生成若干个节点，然后随机对这些节点
进行连接，得到一个随机的graph，然后将subgraph插入产生的随机graph，
得到graph数据。
Input :
    sub_graph : 子图数据
    node_num : 生成随机graph的节点个数
    edge_num : 生成随机graph的边条数
    ins_edge_num : 将subgraph插入随机graph时，连接到随机graph的边条数
'''
def genGraph(sub_graph, node_num, edge_num, ins_edge_num):
    nodes_list = list(np.random.randint(low=1, high=11, size=(node_num)))
    label_list = [-1] * len(nodes_list)
    edges_list = []
    edge_proba = edge_num / (node_num * (node_num - 1) / 2)
    for n in range(node_num-1):
        end_nodes = np.random.choice(a=np.arange(n+1, node_num), 
                                     size=(round(edge_proba * (node_num-1-n))),
                                     replace=False)
        edges_list.extend([(n, e) for e in end_nodes])
    # Insert subgraph
    nodes_list.extend(sub_graph[0])  # add subgraph nodes
    label_list.extend([1]*len(sub_graph[0]))
    head_nodes = np.random.choice(a=np.arange(node_num),
                                  size=(ins_edge_num))
    tail_nodes = np.random.choice(a=np.arange(node_num, len(nodes_list)),
                                  size=(ins_edge_num))
    edges_list.extend([(i,j) for i,j in zip(head_nodes, tail_nodes)])
    edges_list.extend([(i+node_num,j+node_num) for i,j in sub_graph[1]])
    return (nodes_list, edges_list, label_list)
    
'''
Definition of subgraph and dataset generation
'''
np.random.seed(0)
N = 600          # graph dataset length
node_nums = [5, 10, 15, 20]
ins_nums = [4, 8, 12, 16]


sub_graph = ([1,5,5,8],[(0,1),(0,3),(1,3),(2,3)])  # (Node_list, Edge_list)
dataset = []
for i in range(N):
    nnum = random.choice(node_nums)   # node_num
    ins_en = random.choice(ins_nums)# ins_edge_num
    dataset.append(genGraph(sub_graph, nnum, 2*nnum, ins_en))


# ## 构建模型

# ### GCN构建

# In[4]:


'''
同Linear GNN
'''
class AggrSum(nn.Module):
    def __init__(self):
        super(AggrSum, self).__init__()
    
    def forward(self, H, X_node, node_num):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * node_num, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0,node_num-1).float(), 1)
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
    def __init__(self, in_channel, out_channel):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum()
        
    def forward(self, x, edge_index):
        # Add self-connect edges
        edge_index = self.addSelfConnect(edge_index, x.shape[0])
        node_num = x.shape[0]
        
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
        aggr =  self.aggregation(tar_matrix, row, node_num)  # (N, out_channel)
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


# ### 两层GCN构建

# In[5]:


'''
构建模型，使用两层GCN，第一层GCN使得节点特征矩阵
    (N, in_channel) -> (N, out_channel)
第二层GCN直接输出
    (N, out_channel) -> (N, num_class)
激活函数使用relu函数，网络最后对节点的各个类别score使用softmax归一化。
'''
class Net(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16)
        self.conv2 = GCNConv(16, num_class)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.softmax(x, dim=-1)


# ### 准备输入数据

# In[6]:


embedding = np.diag(np.ones((10)))

'''
得到x和edge_index输入
'''
def getInput(graph):
    x = embedding[[i-1 for i in graph[0]]]
    edge_index = np.array([np.array([i,j]) for i,j in graph[1]]).T  # (2, E)
    edge_index = np.concatenate([edge_index] * 2, axis=1)   # (2, 2*E)
    y = np.array(graph[2])
    return x, edge_index, y


# ### 开始训练

# In[ ]:


'''
开始训练模型
'''
device = torch.device('cpu')# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(10, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# loss_fn = nn.CrossEntropyLoss()

'''
模型评估函数
'''
def evalModel(model, dataset):
    for graph in dataset:
        x, edge_index, y = getInput(graph)
        x = torch.from_numpy(x).float()
        edge_index = torch.from_numpy(edge_index).long()
        y[y < 0] = 0
        y = torch.from_numpy(y).long()
        
        acc_list = []
        _, pred = model(x, edge_index).max(dim=1)
        acc_list.append(float(pred.eq(y).sum().item())/y.shape[0])
    return sum(acc_list)/ len(acc_list)

for epoch in range(200):
    for step, graph in enumerate(dataset[:400]):
        # Get input
        x, edge_index, y = getInput(graph)
        x = torch.from_numpy(x).float().to(device)
        edge_index = torch.from_numpy(edge_index).long().to(device)
        y[y < 0] = 0
        y = torch.from_numpy(y).long().to(device)
        
        model.train()
        optimizer.zero_grad()

        # Get output
        out = model(x, edge_index)   # (N, 2)
        # Get loss
        loss = F.cross_entropy(out, y)

        # Backward
        loss.backward()
        optimizer.step()
        
        # Get predictions and calculate training accuracy
        _, pred = out.cpu().detach().max(dim=-1)  # (N)
        y = y.cpu().detach()
        correct = float(pred.eq(y).sum().item())
        acc = correct / pred.shape[0]
        print('[Epoch {}/200, step {}/400] Loss {:.4f}, train acc {:.4f}'.format(epoch, step, loss.cpu().detach().data.item(), acc))

    # Evaluation on test data every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        print('Accuracy: {:.4f}'.format(evalModel(model, dataset[400:])))

