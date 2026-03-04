import torch
import torch.nn as nn

class PhonemeLID(nn.Module):
    def __init__(self, vocab_size=132, embed_dim=128, hidden_dim=256, num_labels=15):
        super(PhonemeLID, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional LSTM with 2 layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # Attention layer over the bi-directional hidden states (hidden_dim * 2)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        # Final classification layer
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        
    def forward(self, x):
        embedded = self.embed(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        
        # Calculate attention weights
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
        # Apply attention to get context vector
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        
        # Final prediction
        out = self.fc(context)
        return out
