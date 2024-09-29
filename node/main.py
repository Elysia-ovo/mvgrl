import argparse
import numpy as np
import torch as th
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

from dataset import process_two_graphs_all
from model import MVGRL, LogReg

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: -1, using cpu.')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')#用于早停（early stopping）的耐心值，默认值为 20。若连续 20 轮损失未降低，则停止训练。
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')#MVGRL 模型的学习率，默认值为 0.001
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')#线性评估器的学习率，默认值为 0.01
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')#MVGRL 模型的权重衰减（正则化）参数，默认值为 0.0
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')#线性评估器的权重衰减参数，默认值为 0.0
parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')#--epsilon：扩散图的边掩码阈值，默认值为 0.01
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')#--hid_dim：隐藏层的维度，默认值为 512

args = parser.parse_args()

#这个地方修改cuda设备，要是爆显存的话改一下
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
#确保以下的代码尽在脚本作为主程序运行的时候执行，而不在被导入时执行
if __name__ == '__main__':
    print(args)
    assembly_graph_file = "/home/zhaozhimiao/Elysia/MyWay/beer_assembly_graph.txt"
    hic_file = "/home/zhaozhimiao/Elysia/MyWay/hic_map.txt"
    feat_file = "/home/zhaozhimiao/Elysia/MyWay/beer_feature.txt"
    # Step 1: Prepare data =================================================================== #
    #原始图结构，差分图结构，节点特征矩阵，节点标签（从哪里来的？直接给的），训练，验证，测试集的节点索引，边权重
    AsGra,HicGra,feat,edge_weight = process_two_graphs_all(assembly_graph_file,hic_file,feat_file)
    print(len(feat))
    if isinstance(edge_weight, np.ndarray):
        edge_weight = th.from_numpy(edge_weight).float().to(args.device)
    # Step 2: 为自环分配权重
    # 获取图中的节点数（自环的数量等于节点数，因为每个节点添加一条自环）
    num_nodes = AsGra.number_of_nodes()
    print("NodeNum",num_nodes)

    # 假设每个自环的权重为 1.0（你也可以根据需要设定不同的权重）
    self_loop_weight = th.ones(num_nodes).float().to(args.device)

    # Step 3: 拼接原始边的权重和自环的权重
    # 使用 torch.cat 拼接两个张量：原有边权重和自环权重
    new_edge_weight = th.cat([edge_weight, self_loop_weight])
    in_degrees = HicGra.in_degrees()
    zero_in_degree_nodes = th.nonzero(in_degrees == 0, as_tuple=False).squeeze()

    # 打印 0 入度节点
    print(f"Number of 0 in-degree nodes: {zero_in_degree_nodes.shape[0]}")
    print("Nodes with 0 in-degree:", zero_in_degree_nodes.tolist())
    #输入特征的维度
    n_feat = feat.shape[1]
    #将图和特征到设备
    AsGra = AsGra.to(args.device)
    HicGra = HicGra.to(args.device)
    feat = feat.to(args.device)
    edge_weight = th.tensor(new_edge_weight).float().to(args.device)
    #图中的节点数量
    n_node = AsGra.number_of_nodes()
    #正样本标签
    lbl1 = th.ones(n_node * 2)
    #负样本标签
    lbl2 = th.zeros(n_node * 2)
    #连接起来，形成一个标签张量
    lbl = th.cat((lbl1, lbl2))

    # Step 2: Create model =================================================================== #
    # 创建模型初始化  第一个参数是输入特征的维度 第二个参数是隐藏层的维度
    model = MVGRL(n_feat, args.hid_dim)
    # 将模型移动到设备
    model = model.to(args.device)
    # 将标签移动到设备（为什么不在step1中移动捏）
    lbl = lbl.to(args.device)

    # Step 3: Create training components 创建训练组件========================================== #
    # 优化器：使用亚当优化器进行优化：第一个参数，模型的可训练参数 第二个参数，学习率第三个参数权重衰减参数
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    # 二元交叉熵损失函数，结合了sigmoid激活函数和二元交叉熵损失 目的是区分正负样本
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs训练过程 ========================================================= #
    best = float('inf')#初始化为正无穷，用于记录当前最低的损失值
    cnt_wait = 0 #记录上一次最佳损失以来的连续未改进轮数
    for epoch in range(args.epochs):
        # 模型设置为训练模式
        model.train()
        #清除模型参数的梯度，避免梯度类加
        optimizer.zero_grad()
        #生成一个 [0, 1, ..., n_node-1] 的随机排列
        shuf_idx = np.random.permutation(n_node)
        #根据打乱的索引重新排列节点特征，生成打乱后的特征张量
        shuf_feat = feat[shuf_idx, :]
        #将打乱后的特征张量移动到指定设备
        shuf_feat = shuf_feat.to(args.device)
        #调用 MVGRL 模型的 forward 方法，传入原始图、差分图、原始特征、打乱特征和边权重
        print(f"Number of edges in graph: {HicGra.number_of_edges()}")
        print(f"Shape of edge_weight: {edge_weight.shape}")
        out = model(AsGra, HicGra, feat, shuf_feat, edge_weight)#这个地方是api特性，默认调用的是前向传播
        # BCEWithLogitsLoss 计算模型输出 out 与标签 lbl 之间的损失
        loss = loss_fn(out, lbl)
        #计算损失相对于模型参数的梯度
        loss.backward()
        #根据计算的梯度更新模型参数
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
        #检查损失是否降低
        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break
    # 加载最佳模型并获取嵌入表示
    model.load_state_dict(th.load('model.pkl'))
    # 获取节点嵌入表示
    embeds = model.get_embedding(AsGra, HicGra, feat, edge_weight)
    # 假设 embeds 是你已经获得的嵌入特征张量
    # 将 embeds 从 PyTorch 张量转换为 NumPy 数组
    embeds_np = embeds.cpu().detach().numpy()

    # Step 1: 对数据进行标准化（DBSCAN对数据尺度比较敏感，因此通常需要进行标准化）
    scaler = StandardScaler()
    embeds_scaled = scaler.fit_transform(embeds_np)

    # Step 2: 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=5, min_samples=380)  # eps 是邻域半径参数，min_samples 是最小样本数
    labels = dbscan.fit_predict(embeds_scaled)
    output_path = '/home/zhaozhimiao/Elysia/mvgrl/Res/res.txt'  # 替换为你想保存的实际路径
    np.savetxt(output_path, labels, fmt='%d')