import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 设置随机种子，保证结果可复现
from TorchCRF import CRF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, num_teams,team_emb_dim,num_layers=1, dropout=0.2,**kwargs):
        super(LSTMClassifier, self).__init__()
        # 嵌入层：将词汇索引转换为密集向量
        # LSTM层
        self.lstm_input_dim = team_emb_dim * 2
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,  # 修正输入维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # ===================== 新增：球队ID的Embedding层 =====================
        # num_teams：球队ID的最大值+1（确保覆盖所有球队ID）
        # team_emb_dim：每个球队ID映射成的向量维度
        self.team1_emb = nn.Embedding(num_embeddings=num_teams, embedding_dim=team_emb_dim)
        self.team2_emb = nn.Embedding(num_embeddings=num_teams, embedding_dim=team_emb_dim)

        # 全连接层：将LSTM输出映射到最终结果
        self.fc = nn.Linear(hidden_dim, 3)

        # 激活函数：sigmoid用于二分类
        self.softmax = nn.Softmax(-1)


    def forward(self, team_a_id, team_b_id, **kwargs):
        """
        前向传播
        Args:
            team_a_id: (batch_size, seq_len) 主队ID序列
            team_b_id: (batch_size, seq_len) 客队ID序列
        Returns:
            out: (batch_size, seq_len, 3) 预测结果（胜/平/负的概率）
        """
        # 通过Embedding层获取球队向量
        # team_a_emb: (batch_size, seq_len, team_emb_dim)
        # team_b_emb: (batch_size, seq_len, team_emb_dim)
        team_a_emb = self.team1_emb(team_a_id)
        team_b_emb = self.team2_emb(team_b_id)
        
        # 拼接两个球队的embedding作为LSTM输入
        # x: (batch_size, seq_len, team_emb_dim * 2)
        x = torch.cat([team_a_emb, team_b_emb], dim=-1)
        
        # LSTM层
        # lstm_out形状: (batch_size, seq_len, hidden_dim)
        # hidden和cell是最后时刻的隐状态和细胞状态
        lstm_out, (hidden, cell) = self.lstm(x)
        # 全连接层和激活函数
        out = self.fc(lstm_out)  # (batch_size, seq_len, 3)
        out = self.softmax(out)  # (batch_size, seq_len, 3)

        return out  # 返回每个时间步的预测结果