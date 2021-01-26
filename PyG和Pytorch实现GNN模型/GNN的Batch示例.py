#!/usr/bin/env python
# coding: utf-8

# # GNN的Batch示例

# **问题简介**：由于GNN处理的数据通常来说是不规则、格式不统一的图(graph)，因此，如何将数据进行批处理并输入到神经网络中进行训练是一个比较常见的问题，该代码使用`对角邻接矩阵`的方式来实现批处理问题(受到了PyG框架的启发)。该代码的数据集使用人工生成的图分类数据集，并使用Pytorch框架进行实现数据载入、模型构建、训练、评估等流程。

# # 任务定义和数据集生成

# ## 任务定义

# 子图匹配分类：给定一个子图(subgraph)$g$以及图(graph)的数据集$\mathcal{G}=\{G_1,G_2,...,G_n\}$，对应的标签为$\mathcal{Y}=\{y_1,y_2,...,y_n\}$，对于任意的图(graph)$G_i$及其标签$y_i$，有：
# $$
# \begin{equation}
# y_i=\left\{
# \begin{aligned}
# 1 & \text{ }G_i包含子图g \\
# 0 & \text{ }G_i不包含子图g \\
# \end{aligned}
# \right.
# \end{equation}
# $$

# ## 数据集生成

# 图(graph)都可以定义为$m$个节点的集合$\mathcal{N}=\{v_1,v_2,...,v_m\}$和$n$条边的集合$\mathcal{E}=\{e_1,e_2,...,e_n\}$，其中，边的数据结构为两个节点的元组，即$(v_i,v_j)$。现设定数据集： 
# + 有26种节点：A,B,C,...,Z。每一种节点都有特定的特征向量，比如one-hot。
# + 图(graph)由不定数量的上述类型节点和不定数量连接的边构成。 

# **注意**：
# + 为了生成代码所需的字典和数据文件，需要运行下面的单元。
#   + 生成训练集：修改第53行的`graph_num=10000`，变量以及第64行的`random.seed(0)`。表示使用随机种子0来生成包含有10000个graph的训练集。
#   + 生成验证集：修改第53行的`graph_num=2000`，变量以及第64行的`random.seed(1)`。表示使用随机种子1来生成包含有2000个graph的验证集。
# + 生成的字典和数据集文件保存在`/PyG和Pytorch实现GNN模型/data/`文件夹下，该文件夹下有三个文件：
# ```
# /PyG和Pytorch实现GNN模型/data/nodes_dict.json
# /PyG和Pytorch实现GNN模型/data/dataset.json
# /PyG和Pytorch实现GNN模型/data/dataset_val.json
# ```

# In[75]:


"""
Code for dataset generation.
"""
import os
import string
import numpy as np
import json
import random
import networkx as nx
from matplotlib import pyplot as plt
import time

"""
Generating nodes dict.
"""
if not os.path.exists("./data/"):
    os.mkdir("./data/")
node_types = list(string.ascii_uppercase)
nodes_dict = dict([(k, v) for v, k in enumerate(node_types)])
nodes_dict_path = "./data/nodes_dict.json"

print("Saving node dict...")
with open(nodes_dict_path, "w") as fp:
    json.dump({
        "itos" : node_types,
        "stoi" : nodes_dict
    }, fp)
print("Successfully saving dict!")

"""
Show graph.
Input : 
    g : tuple(list, list)
"""
def show_graph(g):
    labels = dict([(k,v) for k, v in enumerate(g[0])])
    nodes = range(len(g[0]))
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(g[1])
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, labels)

subgraph = (["A", "A", "B", "C"], 
            [(0, 1),
             (0, 2),
             (1, 2),
             (2, 3)])
min_nodes_num = 5
max_nodes_num = 50
graph_num = 10000

"""
Generating graph dataset. There are three steps:
Step1 : Randomly choose number of nodes(N).
Step2 : Generate random graph with edge number ranging from N-1 to N * (N - 1) / 2.
Step3 : Remove unconnected graph.
Step4 : Add subgraph to some graphs.
"""
N = 0
graphs = []
random.seed(0)
while N < graph_num:
    node_num = random.randint(min_nodes_num, max_nodes_num)
    edge_num = random.randint(node_num-1, node_num * (node_num - 1) / 2)
    G = nx.random_graphs.dense_gnm_random_graph(node_num, edge_num)
    if nx.connected.is_connected(G):
        graphs.append(G)
        N += 1
        if N % 1000 == 0:
            print("{} graphs have been generated!".format(N))

"""
Transform nx.Graph into our graph type.
"""
def transform_nx_graph(g):
    nodes = random.choices(population=node_types, k=len(g.nodes))
    edges = list(g.edges)
    
    return (nodes, edges)

"""
Merge subgraph into graph.
"""
def merge_subgraph_into_graph(g, sg):
    g_node_num = len(g[0])
    sg_node_num = len(sg[0])
    
    merge_edges = [(s+g_node_num, d+g_node_num) for s, d in sg[1]]
    
    g_range = range(g_node_num)
    sg_range = range(g_node_num, g_node_num+sg_node_num)
    new_edges_num = random.randint(1, g_node_num)
    src_list = random.choices(population=g_range, k=new_edges_num)
    dst_list = random.choices(population=sg_range, k=new_edges_num)
    new_edges = [(s, d) for s, d in zip(src_list, dst_list)]
    new_edges = list(set(new_edges))
    
    merge_graph = (g[0]+sg[0],
                   g[1]+merge_edges+new_edges)
    
    return merge_graph

graphs = [transform_nx_graph(g) for g in graphs]
graphs_with_sg = [(merge_subgraph_into_graph(g, subgraph), 1) for g in graphs[:len(graphs)//2]]
graphs_without_sg = [(g, 0) for g in graphs[len(graphs)//2:]]
graphs = graphs_with_sg + graphs_without_sg
random.shuffle(graphs)

print("Saving dataset!")
dataset_path = "./data/dataset_val.json"
with open(dataset_path, "w") as fp:
    json.dump({
        "time" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "subgraph" : subgraph,
        "graphs" : graphs,
        "min_nodes_num" : min_nodes_num,
        "max_nodes_num" : max_nodes_num,
        "graphs_num" : graph_num
    }, fp)


# # Dataloader定义 

# In[71]:


import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset

dict_path = "./data/nodes_dict.json"
graph_path = "./data/dataset.json"

class GraphDataset(Dataset):
    def __init__(self, dict_path, graph_path):
        super(GraphDataset, self).__init__()
        assert os.path.exists(dict_path) and os.path.exists(graph_path)
        
        with open(dict_path, "r") as fp:
            self.node_dict = json.load(fp)["stoi"]
        with open(graph_path, "r") as fp:
            self.graphs = json.load(fp)
    
    def __len__(self):
        return len(self.graphs["graphs"])
    
    def __getitem__(self, ind):
        graph, label = self.graphs["graphs"][ind]
        nodes = [self.node_dict[n] for n in graph[0]]
        edges = [list(e) for e in graph[1]]
        
        return np.array(nodes, dtype=np.int64), np.array(edges, dtype=np.int64), np.array([label], dtype=np.int64)

"""
Transform edges into adjacency matrix.
Input :
    node_num : Total num of nodes.
    edges : Edges matrix.
Output :
    m : Adjacency matrix.
"""
def edges_to_matrix(node_num, edges):
    m = np.zeros(shape=(node_num, node_num), dtype=np.uint8)
    m[edges[:,0], edges[:,1]] = 1
    m[edges[:,1], edges[:,0]] = 1
    m[np.arange(node_num), np.arange(node_num)] = 1
    
    return m

"""
Combine multiple graphs into one large graph.
"""
def collate_fn(batch):
    nodes_list = [b[0] for b in batch]
    nodes = np.concatenate(nodes_list, axis=0)
    
    nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
    nodes_inds = np.cumsum(nodes_lens)
    nodes_num = nodes_inds[-1]
    nodes_inds = np.insert(nodes_inds, 0, 0)
    nodes_inds = np.delete(nodes_inds, -1)
    edges_list = [b[1] for b in batch]
    edges_list = [e+i for e,i in zip(edges_list, nodes_inds)]
    edges = np.concatenate(edges_list, axis=0)
    m = edges_to_matrix(nodes_num, edges)
    
    labels = [b[2] for b in batch]
    labels = np.concatenate(labels, axis=0)
    
    batch_mask = [np.array([i]*k, dtype=np.int32) for i, k in zip(range(len(batch)), nodes_lens)]
    batch_mask = np.concatenate(batch_mask, axis=0)
    return torch.from_numpy(nodes), torch.from_numpy(m).float(), torch.from_numpy(labels), torch.from_numpy(batch_mask)


# # GCN模型构建

# 根据PyG文档中对GCN的公式表示：
# $$
# x_i^{(k)}=\sum_{j\in{}\mathcal{N(i)}\cup{}\{i\}}\frac{1}{\sqrt{\textrm{deg}(i)}\cdot{}\sqrt{\textrm{deg}(j)}}\cdot{}(\Theta\cdot{}x_j^{(k-1)})
# $$

# In[72]:


import torch
import torch.nn as nn

"""
Implement of GCN module.
"""
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super(GCN, self).__init__()
        self.trans_msg = nn.Linear(in_dim, out_dim)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    """
    Input : 
        x : (N, in_dim)
        m : (N, N)
    Output :
        out : (N, out_dim)
    """
    def forward(self, x:torch.Tensor, m:torch.Tensor):
        x_msg = self.trans_msg(x)
        x_msg = self.nonlinear(x_msg)
        x_msg = self.dropout(x_msg)
        
        row_degree = torch.sum(m, dim=1, keepdim=True)   # (N, 1)
        col_degree = torch.sum(m, dim=0, keepdim=True)   # (1, N)
        degree = torch.mm(torch.sqrt(row_degree), torch.sqrt(col_degree))  # (N, N)
        out = torch.mm(m / degree, x_msg)
        
        return out

"""
Implement of GCN network.
"""
class Net(nn.Module):
    def __init__(self, nodes_num, embedding_dim, hidden_dims, num_classes, dropout=0.):
        super(Net, self).__init__()
        
        self.node_embedding = nn.Embedding(nodes_num, embedding_dim)
        gcns = []
        in_dim = embedding_dim
        for d in hidden_dims:
            gcns.append(GCN(in_dim, d, dropout))
            in_dim = d
        self.gcns = nn.ModuleList(gcns)
        
        self.classifier = nn.Linear(in_dim, num_classes)
    
    """
    Input :
        x : (N, out_dim)
        bm : (N, )
    Output :
        out : (batch_size, out_dim)
    """
    def gcn_maxpooling(self, x, bm):
        batch_size = torch.max(bm)+1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:,0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.max(x_ind, dim=0, keepdim=False)[0])
        out = torch.stack(out, dim=0)
        
        return out
    
    def gcn_meanpooling(self, x, lens):
        batch_size = torch.max(bm)+1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:,0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.mean(x_ind, dim=0, keepdim=False))
        out = torch.stack(out, dim=0)
        
        return out
    
    def gcn_sumpooling(self, x, lens):
        batch_size = torch.max(bm)+1
        out = []
        for i in range(batch_size):
            inds = (bm == i).nonzero()[:,0]
            x_ind = torch.index_select(x, dim=0, index=inds)
            out.append(torch.sum(x_ind, dim=0, keepdim=False))
        out = torch.stack(out, dim=0)
        
        return out
    
    """
    Input :
        x : (N, )
        m : (N, N)
        bm : (N, )
    Output :
        output : (batch_size, num_classes)
    """
    def forward(self, x, m, bm):
        x_emb = self.node_embedding(x)  # (N, embedding_dim)
        out = x_emb
        for ml in self.gcns:
            out = ml(out, m)
        output = self.gcn_maxpooling(out, bm)   # (batch_size, out_dim)
        
        logits = self.classifier(output)  # (batch_size, num_classes)
        
        return logits


# # 模型训练

# **注意**：先运行`Dataloader定义`和`GCN模型构建`单元。

# In[ ]:


model = Net(nodes_num=len(ds.node_dict), embedding_dim=20, hidden_dims=[32], num_classes=2, dropout=0.2)
model = model.cuda() if torch.cuda.is_available() else model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
batch_size = 5

ds = GraphDataset(dict_path, graph_path)
dataloader = DataLoader(dataset=ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn,
                        drop_last=False)

ds_val = GraphDataset(dict_path, "./data/dataset_val.json")
dataloader_val = DataLoader(dataset=ds_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn,
                            drop_last=False)

def eval_model(dataloader, model):
    right = 0
    total = 0
    for n, m, l, bm in dataloader:
        n = n.cuda()
        m = m.cuda()
        l = l.cuda()
        bm = bm.cuda()
        
        logits = model(n, m, bm)
        
        preds = torch.max(logits, dim=-1, keepdim=False)[1] == l
        right += torch.sum(preds).cpu().detach().data
        total += preds.shape[0]
    
    return float(right) / total

epochs = 20
steps = len(ds) // batch_size
loss_fn = torch.nn.CrossEntropyLoss()

for ep in range(epochs):
    step = 0
    for n, m, l, bm in dataloader:
        n = n.cuda()
        m = m.cuda()
        l = l.cuda()
        bm = bm.cuda()
        
        logits = model(n, m, bm)
        
        loss = loss_fn(logits, l)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("Epoch {}, step {}/{} : loss {}".format(ep, step, steps, loss.detach().cpu().data))
        step += 1
    model.eval()
    acc = eval_model(dataloader_val, model)
    print("Epoch {}, val accuracy : {:.4f}".format(ep, acc))
    model.train()

