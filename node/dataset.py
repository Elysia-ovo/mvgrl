''' Code adapted from https://github.com/kavehhassani/mvgrl '''
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from dgl.nn import APPNPConv

def preprocess_features(features):
    """特征预处理函数

    Parameters
    ----------
    features
        特征

    Returns
    -------
        如果特征是numpy数组，直接返回。如果是稀疏矩阵，返回密集矩阵和稀疏矩阵的元组表示。
    """
    # Row-normalize feature matrix and convert to tuple representation
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """将稀疏矩阵转换为元组表示，包含坐标、值和形状。这种表示方式在某些图神经网络的实现中更为高效。

    Parameters
    ----------
    sparse_mx
        
    """

    def to_tuple(mx):
        """把非稀疏矩阵转换成稀疏矩阵

        Parameters
        ----------
        mx
            输入的矩阵

        Returns
        -------
            三元组
        """
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    """PageRank算法

    Parameters
    ----------
    graph
        输入的图，使用network.graph
    alpha, optional
        阻尼稀疏控制着随机游走中，返回到初始节点的概率。值越大，返回的概率越大  
    self_loop, optional
        是否在图中添加自环。自环可以在某些场景下增加稳定性，常见于图神经网络中的归一化操作。

    Returns
    -------
        函数返回的是一个矩阵，它表示图中每个节点到其他节点的个性化 PageRank 值，衡量了节点之间通过随机游走的可达性。
    """
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def process_dataset(name, epsilon):
    """加载图神经网络（Graph Neural Network, GNN）常用的两个数据集（cora 或 citeseer），对数据集进行处理并生成两个图结构（原始图和基于 Personalized PageRank (PPR) 的差分图），同时返回节点特征、标签以及训练、验证和测试数据的索引。

    Parameters
    ----------
    name
        指定要处理的数据集名称
    epsilon
        稀疏化的阈值，用于过滤差分邻接矩阵中的小值。小于 epsilon 的值将被设为 0。

    Returns
    -------
        返回节点特征、标签以及训练、验证和测试数据的索引
    """
    #cora数据集没有任何的其他的处理直接
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()

    graph = dataset[0]#这个地方提取出来就是DGL结构的图
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    nx_g = dgl.to_networkx(graph)

    print('computing ppr')
    diff_adj = compute_ppr(nx_g, 0.2)#所以事实上扩散完成后的矩阵是差分矩阵就是使用这个ppr计算出来的
    print('computing end')

    if name == 'citeseer':
        print('additional processing')
        feat = th.tensor(preprocess_features(feat.numpy())).float()
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    graph = graph.add_self_loop()

    return graph, diff_graph, feat
def process_two_graphs_without_isolate(txt_file1, txt_file2, feat_file, epsilon=0):
    """
    读取一个无权图和一个有权图的边信息，计算两者的节点并集，构建图结构，并加载节点特征。
    删除没有边连接的节点及其特征。

    Parameters
    ----------
    txt_file1 : str
        无权图的边信息文件路径
    txt_file2 : str
        有权图的边信息文件路径
    feat_file : str
        节点特征文件路径
    epsilon : float, optional
        稀疏化阈值，用于过滤有权图中小于 epsilon 的边权重

    Returns
    -------
    g1 : DGLGraph
        无权图的 DGL 表示，基于节点并集
    g2 : DGLGraph
        有权图的 DGL 表示，基于节点并集
    filtered_feat : torch.Tensor
        有效节点的特征，移除了孤立节点
    """
    
    def build_adj_matrix(txt_file, total_nodes, weighted=False):
        """根据 txt 文件构建邻接矩阵，支持无权图和有权图"""
        adj_matrix = np.zeros((total_nodes, total_nodes))
        edge_weights = None
        if weighted:
            edge_weights = np.zeros((total_nodes, total_nodes))  # 保存边权重的矩阵

        with open(txt_file, 'r') as f:
            for line in f:
                if weighted:
                    u, v, w = map(float, line.split())  # 有权图读取边权重
                    u, v = int(u), int(v)  # 转换为整数节点索引
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
                    edge_weights[u, v] = w  # 保存边权重
                    edge_weights[v, u] = w  # 对称处理
                else:
                    u, v = map(int, line.split())  # 无权图不读取权重
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
        return adj_matrix, edge_weights
    
    # Step 1: 读取两个图的边信息，计算节点的并集
    edges1 = np.loadtxt(txt_file1, dtype=int)  # 无权图边信息
    edges2 = np.loadtxt(txt_file2)  # 有权图边信息

    # 找到两张图的最大节点编号，以此确定总的节点数量
    max_node1 = max(edges1.max(), edges2[:, :2].astype(int).max())
    total_nodes = max_node1 + 1  # 节点编号从 0 开始，所以总数要加 1
    print(total_nodes)

    # Step 2: 构建两张图的邻接矩阵
    adj_matrix1, _ = build_adj_matrix(txt_file1, total_nodes, weighted=False)  # 无权图
    print("adj_matrix1",len(adj_matrix1))
    adj_matrix2, edge_weights2 = build_adj_matrix(txt_file2, total_nodes, weighted=True)  # 有权图

    # Step 3: 稀疏化有权图的邻接矩阵
    if epsilon > 0:
        adj_matrix2[edge_weights2 < epsilon] = 0
        edge_weights2[edge_weights2 < epsilon] = 0

    # Step 4: 找到两个图中有边连接的节点
    src1, dst1 = np.nonzero(adj_matrix1)
    src2, dst2 = np.nonzero(adj_matrix2)
    # print(len(src1))
    # print(len(dst1))

    # 获取所有有连接的节点的并集
    connected_nodes = np.unique(np.concatenate((src1, dst1, src2, dst2)))
    print("所有节点的个数connectnode交集",len(connected_nodes))
    # Step 5: 构建无权图和有权图
    g1 = dgl.graph((src1, dst1), num_nodes=total_nodes)  # 无权图
    g2 = dgl.graph((src2, dst2), num_nodes=total_nodes)  # 有权图

    # 将边权重添加到有权图的边数据中
    edge_weight_indices = np.nonzero(edge_weights2)
    edge_weight_values = edge_weights2[edge_weight_indices]
    # g2.edata['weight'] = th.tensor(edge_weight_values, dtype=th.float32)

    # Step 6: 加载节点特征
    feat = np.loadtxt(feat_file)

    # 检查特征矩阵大小
    if feat.shape[0] != total_nodes:
        raise ValueError(f"特征文件中的节点数量 ({feat.shape[0]}) 和总节点数量 ({total_nodes}) 不匹配。")

    # 仅保留有连接的节点的特征
    filtered_feat = feat[connected_nodes]
    filtered_feat = th.tensor(filtered_feat, dtype=th.float32)

    # 将图也调整为只包含有连接的节点
    g1 = dgl.node_subgraph(g1, connected_nodes)
    g2 = dgl.node_subgraph(g2, connected_nodes)
    g1 = g1.add_self_loop()
    g2 = g2.add_self_loop()
    return g1, g2, filtered_feat,edge_weight_values
def process_dataset_appnp(epsilon):
    """处理 Pubmed 数据集，使用 APPNP（Approximate Personalized Propagation of Neural Predictions） 算法生成一个基于 APPNP 的差分图，并返回经过处理的图、差分图、特征、标签以及数据集的划分（训练、验证、测试）的索引。

    Parameters
    ----------
    epsilon
        用于对邻接矩阵的稀疏化。稀疏化过程会将 PPR 矩阵中的小于 epsilon 的值设为 0，从而使得矩阵变得稀疏，减少计算复杂度。

    Returns
    -------
        graph：经过处理的原始图，已经添加了自环。
        diff_graph：基于 APPNP 算法生成的差分图。
        feat：节点特征矩阵。
        label：节点标签。
        train_idx, val_idx, test_idx：训练、验证和测试集节点的索引。
        diff_weight：差分图中边的权重。
    """
    k = 20
    alpha = 0.2
    dataset = PubmedGraphDataset()
    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    appnp = APPNPConv(k, alpha)
    id = th.eye(graph.number_of_nodes()).float()
    diff_adj = appnp(graph.add_self_loop(), id).numpy()

    diff_adj[diff_adj < epsilon] = 0
    scaler = MinMaxScaler()
    scaler.fit(diff_adj)
    diff_adj = scaler.transform(diff_adj)
    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    graph = dgl.add_self_loop(graph)
    return graph, diff_graph, feat, label, train_idx, val_idx, test_idx, diff_weight
def process_two_graphs_all(txt_file1, txt_file2, feat_file, epsilon=0):
    """
    读取一个无权图和一个有权图的边信息，计算两者的节点并集，构建图结构，并加载节点特征。
    删除没有边连接的节点及其特征。

    Parameters
    ----------
    txt_file1 : str
        无权图的边信息文件路径
    txt_file2 : str
        有权图的边信息文件路径
    feat_file : str
        节点特征文件路径
    epsilon : float, optional
        稀疏化阈值，用于过滤有权图中小于 epsilon 的边权重

    Returns
    -------
    g1 : DGLGraph
        无权图的 DGL 表示，基于节点并集
    g2 : DGLGraph
        有权图的 DGL 表示，基于节点并集
    filtered_feat : torch.Tensor
        有效节点的特征，移除了孤立节点
    """
    
    def build_adj_matrix(txt_file, total_nodes, weighted=False):
        """根据 txt 文件构建邻接矩阵，支持无权图和有权图"""
        adj_matrix = np.zeros((total_nodes, total_nodes))
        edge_weights = None
        if weighted:
            edge_weights = np.zeros((total_nodes, total_nodes))  # 保存边权重的矩阵

        with open(txt_file, 'r') as f:
            for line in f:
                if weighted:
                    u, v, w = map(float, line.split())  # 有权图读取边权重
                    u, v = int(u), int(v)  # 转换为整数节点索引
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
                    edge_weights[u, v] = w  # 保存边权重
                    edge_weights[v, u] = w  # 对称处理
                else:
                    u, v = map(int, line.split())  # 无权图不读取权重
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1
        return adj_matrix, edge_weights
    
    # Step 1: 读取两个图的边信息，计算节点的并集
    edges1 = np.loadtxt(txt_file1, dtype=int)  # 无权图边信息
    edges2 = np.loadtxt(txt_file2)  # 有权图边信息

    # 找到两张图的最大节点编号，以此确定总的节点数量
    max_node1 = max(edges1.max(), edges2[:, :2].astype(int).max())
    total_nodes = max_node1 + 1  # 节点编号从 0 开始，所以总数要加 1
    print(total_nodes)

    # Step 2: 构建两张图的邻接矩阵
    adj_matrix1, _ = build_adj_matrix(txt_file1, total_nodes, weighted=False)  # 无权图
    print("adj_matrix1",len(adj_matrix1))
    adj_matrix2, edge_weights2 = build_adj_matrix(txt_file2, total_nodes, weighted=True)  # 有权图

    # Step 3: 稀疏化有权图的邻接矩阵
    if epsilon > 0:
        adj_matrix2[edge_weights2 < epsilon] = 0
        edge_weights2[edge_weights2 < epsilon] = 0

    # Step 4: 找到两个图中有边连接的节点
    src1, dst1 = np.nonzero(adj_matrix1)
    src2, dst2 = np.nonzero(adj_matrix2)
    # print(len(src1))
    # print(len(dst1))
    # Step 5: 构建无权图和有权图
    g1 = dgl.graph((src1, dst1), num_nodes=total_nodes)  # 无权图
    g2 = dgl.graph((src2, dst2), num_nodes=total_nodes)  # 有权图

    # 将边权重添加到有权图的边数据中
    edge_weight_indices = np.nonzero(edge_weights2)
    edge_weight_values = edge_weights2[edge_weight_indices]
    # g2.edata['weight'] = th.tensor(edge_weight_values, dtype=th.float32)

    # Step 6: 加载节点特征
    feat = np.loadtxt(feat_file)
    feat = th.tensor(feat,dtype = th.float32)
    # 检查特征矩阵大小
    if feat.shape[0] != total_nodes:
        raise ValueError(f"特征文件中的节点数量 ({feat.shape[0]}) 和总节点数量 ({total_nodes}) 不匹配。")
    g1 = g1.add_self_loop()
    g2 = g2.add_self_loop()
    return g1, g2, feat,edge_weight_values
