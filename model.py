import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowPCC(nn.Module):
    def __init__(self, window_length, stride):
        super(SlidingWindowPCC, self).__init__()
        self.window_length = window_length
        self.stride = stride

    def forward(self, input_data):
        batch_size, num_time_points, num_features = input_data.size()
        windows = []
        for i in range(0, num_time_points - self.window_length + 1, self.stride):
            window_data = input_data[:, i:i+self.window_length, :]
            windows.append(window_data)
        windows = torch.stack(windows, dim=1)  # Shape: (batch_size, num_windows, window_length, num_features)

        # Calculate Pearson correlation coefficient
        pccs = []
        for window_data in windows.unbind(1):
            pcc = torch.matmul(window_data.transpose(1, 2), window_data) / window_data.size(1)  # Pairwise correlation
            pccs.append(pcc)
        pccs = torch.stack(pccs, dim=1)  # Shape: (batch_size, num_windows, num_features, num_features)
        return pccs

class GraphAttentionPooling(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphAttentionPooling, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_data):
        # input_data shape: (batch_size, num_windows, num_features, num_features)
        batch_size, num_windows, num_features, _ = input_data.size()
        pooled_data = self.linear(input_data)  # Apply linear transformation
        pooled_data = F.softmax(pooled_data, dim=1)  # Apply softmax to get attention weights
        pooled_data = torch.matmul(pooled_data.transpose(2, 3), input_data)  # Apply attention pooling
        return pooled_data

class DualTemporalGraphLearning(nn.Module):
    def __init__(self, srl_input_size, srl_hidden_size, gcn_input_size, gcn_hidden_size, lstm_hidden_size, num_classes):
        super(DualTemporalGraphLearning, self).__init__()
        self.srl = SRLEncoding(srl_input_size, srl_hidden_size)
        self.gcn = TemporalGraphRepresentationLearning(gcn_input_size, gcn_hidden_size, lstm_hidden_size)
        self.linear = nn.Linear(lstm_hidden_size + gcn_hidden_size, num_classes)

    def forward(self, raw_bold_signals, coarsened_graphs):
        srl_output = self.srl(raw_bold_signals)
        gcn_output = self.gcn(coarsened_graphs)
        fused_output = torch.cat((srl_output, gcn_output), dim=1)  # Concatenate outputs
        output = self.linear(fused_output)  # Linear layer for classification
        return output

class SRLEncoding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SRLEncoding, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, raw_bold_signals):
        return F.relu(self.linear(raw_bold_signals))

class TemporalGraphRepresentationLearning(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_hidden_size):
        super(TemporalGraphRepresentationLearning, self).__init__()
        self.gcn = GCN(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)

    def forward(self, coarsened_graphs):
        gcn_output = self.gcn(coarsened_graphs)
        _, (h_n, _) = self.lstm(gcn_output)  # Apply LSTM
        lstm_output = h_n[-1]  # Extract final hidden state
        return lstm_output

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, coarsened_graphs):
        return F.relu(self.linear(coarsened_graphs))
