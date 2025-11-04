import torch
import torch.nn as nn

class DynamicCleavageRNN(nn.Module):
    """
    动态糖链切割位点预测模型
    
    参数:
    - vocab_size: 词汇表大小（糖单元的种类数）
    - embedding_dim: 嵌入层维度
    - hidden_dim: RNN隐藏层维度
    - num_layers: RNN层数
    - dropout: Dropout比率
    - bidirectional: 是否使用双向RNN
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim=64, 
                 hidden_dim=128, 
                 num_layers=2,
                 dropout=0.3,
                 bidirectional=True):
        super().__init__()
        
        # 保存参数
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,  # +1 for padding
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 分类器
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.junction_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x, mask):
        """
        前向传播
        
        参数:
        - x: 输入序列，形状为 [batch_size, seq_len]
        - mask: 掩码矩阵，形状为 [batch_size, max_junctions]
        
        返回:
        - logits: 预测结果，形状为 [batch_size, max_junctions]
        """
        # 1. 嵌入层
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 2. RNN层
        rnn_out, _ = self.rnn(embeds)  # [batch_size, seq_len, hidden_dim*2]
        
        # 3. 生成连接点特征
        batch_size, seq_len, hidden_size = rnn_out.shape
        junctions = []
        
        # 对每个相邻单元对生成预测
        for i in range(seq_len - 1):
            # 获取相邻单元的特征
            front = rnn_out[:, i, :]
            rear = rnn_out[:, i+1, :]
            
            # 拼接特征
            features = torch.cat([front, rear], dim=1)
            
            # 应用分类器
            junction_pred = self.junction_classifier(features)
            junctions.append(junction_pred)
        
        # 4. 整合所有预测
        if junctions:
            logits = torch.stack(junctions, dim=1)  # [batch, junctions, 2]
            logits = logits.view(batch_size, -1)    # [batch, 2*junctions]
        else:
            # 处理边界情况：序列长度为1
            logits = torch.zeros(batch_size, 0, device=x.device)
        
        # 5. 应用mask - 使用masked_fill更明确地处理无效位置
        if logits.numel() > 0:  # 确保logits不为空
            # 将无效位置设为一个很小的负值，这样sigmoid后接近0
            masked_logits = logits.masked_fill(mask == 0, -1e9)
            return masked_logits
        else:
            return logits
    
    def predict(self, x, mask, threshold=0.5):
        """
        预测函数
        
        参数:
        - x: 输入序列
        - mask: 掩码矩阵
        - threshold: 分类阈值
        
        返回:
        - predictions: 二值预测结果
        """
        self.eval()
        with torch.no_grad():
            logits = self(x, mask)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        return predictions
    
    def save(self, path):
        """
        保存模型
        
        参数:
        - path: 保存路径
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional
            }
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """
        加载模型
        
        参数:
        - path: 模型文件路径
        - device: 设备类型（'cpu'或'cuda'）
        
        返回:
        - model: 加载的模型
        """
        save_dict = torch.load(path, map_location=device)
        model_config = save_dict['model_config']
        
        # 创建模型实例
        model = cls(**model_config)
        
        # 加载模型参数
        model.load_state_dict(save_dict['model_state_dict'])
        
        return model.to(device) 