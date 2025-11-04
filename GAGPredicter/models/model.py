import torch
import torch.nn as nn

class DynamicCleavageRNN(nn.Module):
    """
    Dynamic Glycan Chain Cleavage Site Prediction Model
    
    Parameters:
    - vocab_size: Vocabulary size (number of sugar unit types)
    - embedding_dim: Embedding layer dimension
    - hidden_dim: RNN hidden layer dimension
    - num_layers: Number of RNN layers
    - dropout: Dropout rate
    - bidirectional: Whether to use bidirectional RNN
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim=64, 
                 hidden_dim=128, 
                 num_layers=2,
                 dropout=0.3,
                 bidirectional=True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,  # +1 for padding
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.junction_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x, mask):
        """
        Forward propagation
        
        Parameters:
        - x: Input sequence, shape [batch_size, seq_len]
        - mask: Mask matrix, shape [batch_size, max_junctions]
        
        Returns:
        - logits: Prediction results, shape [batch_size, max_junctions]
        """
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        rnn_out, _ = self.rnn(embeds)  # [batch_size, seq_len, hidden_dim*2]
        
        batch_size, seq_len, hidden_size = rnn_out.shape
        junctions = []
        
        for i in range(seq_len - 1):
            front = rnn_out[:, i, :]
            rear = rnn_out[:, i+1, :]
            
            features = torch.cat([front, rear], dim=1)
            
            junction_pred = self.junction_classifier(features)
            junctions.append(junction_pred)
        
        if junctions:
            logits = torch.stack(junctions, dim=1)  # [batch, junctions, 2]
            logits = logits.view(batch_size, -1)    # [batch, 2*junctions]
        else:
            logits = torch.zeros(batch_size, 0, device=x.device)
        
        if logits.numel() > 0:
            masked_logits = logits.masked_fill(mask == 0, -1e9)
            return masked_logits
        else:
            return logits
    
    def predict(self, x, mask, threshold=0.5):
        """
        Prediction function
        
        Parameters:
        - x: Input sequence
        - mask: Mask matrix
        - threshold: Classification threshold
        
        Returns:
        - predictions: Binary prediction results
        """
        self.eval()
        with torch.no_grad():
            logits = self(x, mask)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        return predictions
    
    def save(self, path):
        """
        Save model
        
        Parameters:
        - path: Save path
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
        Load model
        
        Parameters:
        - path: Model file path
        - device: Device type ('cpu' or 'cuda')
        
        Returns:
        - model: Loaded model
        """
        save_dict = torch.load(path, map_location=device)
        model_config = save_dict['model_config']
        
        model = cls(**model_config)
        
        model.load_state_dict(save_dict['model_state_dict'])
        
        return model.to(device)
