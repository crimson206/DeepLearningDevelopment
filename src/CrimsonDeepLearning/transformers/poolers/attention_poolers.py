import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooler(nn.Module):
    def __init__(self, input_dim, n_hidden):
        super(AttentionPooler, self).__init__()
        # Define fully connected layers for query, key, and value
        self.fc_query = nn.Linear(input_dim, n_hidden)
        self.fc_key = nn.Linear(input_dim, n_hidden)
        self.fc_value = nn.Linear(input_dim, n_hidden)
        self.scale = torch.sqrt(torch.tensor(n_hidden, dtype=torch.float))

    def forward(self, x):
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)

        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)

        weighted_values = torch.bmm(attention_scores, value)

        pooled_output = weighted_values.mean(dim=1, keepdim=True)

        return pooled_output.squeeze()
    

class ConvPooling(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvPooling, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel_size, padding="same")

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze(1)

class FCPooling(nn.Module):
    def __init__(self, n_seq, output_dim):
        super(FCPooling, self).__init__()
        self.fc = nn.Linear(n_seq, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x.permute(0, 2, 1).squeeze()