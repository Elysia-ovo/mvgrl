import unittest
import numpy as np
import torch as th
import dgl
import os
from tempfile import NamedTemporaryFile
from dataset import process_two_graphs  # 假设函数在 your_module 中

class TestProcessTwoGraphs(unittest.TestCase):
    
    def setUp(self):
        # 创建测试用的无权图和有权图边信息，以及特征文件
        
        # 无权图：节点 0-1, 1-2, 2-3
        self.txt_file1 = NamedTemporaryFile(delete=False, mode='w')
        self.txt_file1.write("0 1\n1 2\n2 3\n")
        self.txt_file1.close()

        # 有权图：节点 0-2 (权重 0.5), 1-3 (权重 0.8), 2-3 (权重 1.2)
        self.txt_file2 = NamedTemporaryFile(delete=False, mode='w')
        self.txt_file2.write("0 2 0.5\n1 3 0.8\n2 3 1.2\n")
        self.txt_file2.close()

        # 节点特征: 4 个节点，每个节点有 3 个特征
        self.feat_file = NamedTemporaryFile(delete=False, mode='w')
        np.savetxt(self.feat_file.name, np.array([[1.0, 0.5, 0.2],
                                                  [0.9, 0.1, 0.3],
                                                  [0.4, 0.8, 0.7],
                                                  [0.2, 0.3, 0.9]]))
        self.feat_file.close()

    def tearDown(self):
        # 删除临时文件
        os.remove(self.txt_file1.name)
        os.remove(self.txt_file2.name)
        os.remove(self.feat_file.name)

    def test_process_two_graphs(self):
        # 调用 process_two_graphs 函数
        g1, g2, filtered_feat = process_two_graphs(self.txt_file1.name, self.txt_file2.name, self.feat_file.name)

        # 检查图 g1 和 g2
        self.assertEqual(g1.num_nodes(), 4)  # 总共有 4 个节点
        self.assertEqual(g2.num_nodes(), 4)

        # 检查无权图边数
        self.assertEqual(g1.num_edges(), 3)  # 无权图应有 3 条边

        # 检查有权图边数
        self.assertEqual(g2.num_edges(), 3)  # 有权图应有 3 条边
        
        # 检查有权图的边权重
        edge_weights = g2.edata['weight'].numpy()
        expected_weights = np.array([0.5, 0.8, 1.2])
        np.testing.assert_array_almost_equal(edge_weights, expected_weights)

        # 检查过滤后的节点特征
        expected_feat = np.array([[1.0, 0.5, 0.2],
                                  [0.9, 0.1, 0.3],
                                  [0.4, 0.8, 0.7],
                                  [0.2, 0.3, 0.9]])
        np.testing.assert_array_almost_equal(filtered_feat.numpy(), expected_feat)

    def test_process_two_graphs_with_epsilon(self):
        # 测试带 epsilon 参数的情况，稀疏化有权图
        g1, g2, filtered_feat = process_two_graphs(self.txt_file1.name, self.txt_file2.name, self.feat_file.name, epsilon=1.0)

        # 检查图 g1 和 g2
        self.assertEqual(g1.num_nodes(), 4)  # 仍然是 4 个节点
        self.assertEqual(g2.num_nodes(), 4)

        # 检查有权图边数
        self.assertEqual(g2.num_edges(), 1)  # 因为 epsilon=1.0，只有权重 > 1.0 的边被保留

        # 检查有权图的边权重
        edge_weights = g2.edata['weight'].numpy()
        expected_weights = np.array([1.2])  # 只有权重为 1.2 的边被保留
        np.testing.assert_array_almost_equal(edge_weights, expected_weights)

if __name__ == '__main__':
    unittest.main()
