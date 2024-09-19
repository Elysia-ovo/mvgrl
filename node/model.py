import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling

class LogReg(nn.Module):
    #逻辑回归模型
    def __init__(self, hid_dim, n_classes):
        """构造

        Parameters
        ----------
        hid_dim
            隐藏层函数
        n_classes
            类数
        """
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(nn.Module):
    """_summary_

    Parameters
    ----------
    nn
        返回合并后的评分张量
    """
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)
        #这个双线性层将两个dim维的输入向量映射到一个标量输出

    def forward(self, h1, h2, h3, h4, c1, c2):
        """前向传播

        Parameters
        ----------
        h1
            特征向量
        h2
            特征
        h3
            特征
        h4
            特征
        c1
            条件向
        c2
            条件向量

        Returns
        -------
            
        """
        #拓展上下文向量
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # 计算正样本评分
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # 计算负样本评分
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)
        #合并张亮
        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits

class MVGRL(nn.Module):

    def __init__(self, in_dim, out_dim):
        """构造函数

        Parameters
        ----------
        in_dim
            输入特征的维度，每个节点的特征数量
        out_dim
            输出特征的维度，每个节点在编码后得到的特征数量
        """
        super(MVGRL, self).__init__()
        #两个图卷积层
        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        #一个池化层
        self.pooling = AvgPooling()
        #上面定义的判别器类，用于计算节点之间的表示相似性
        self.disc = Discriminator(out_dim)
        #激活函数
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, feat, edge_weight):
        """前向传播

        Parameters
        ----------
        graph
            原始图结构
        diff_graph
            扰动后的图结构
        feat
            节点的特征矩阵
        edge_weight
            边权重

        Returns
        -------
            节点的嵌入表示
        """
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()#使用detach把嵌入从计算图里面分离，避免后续计算中更新就编码器的参数

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight):
        #h1是通过对原始图和原始特征进行编码，得到的节点表示
        h1 = self.encoder1(graph, feat)
        #h2是通过对扰动图和原始特征进行编码
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)
        #h3是通过原始图个打乱之后的特征进行编码
        h3 = self.encoder1(graph, shuf_feat)
        #扰动图和打乱之后的特征进行编码
        h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)
        #括号里面是h1和h2的全局平均池化，得到的是图的全局表示
        #对得到的全局表示应用sigmoid激活函数，映射到01区间
        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))
        #计算相似性评分
        out = self.disc(h1, h2, h3, h4, c1, c2)
        #返回判别器的评分输出，用于后续的损失计算和模型训练
        return out