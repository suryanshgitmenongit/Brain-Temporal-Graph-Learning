import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.attention_weights = nn.Linear(output_size, 1)

    def forward(self, input_data):
        # input_data shape: (batch_size, num_windows, num_features, num_features)
        batch_size, num_windows, num_features, _ = input_data.size()
        pooled_data = self.linear(input_data)  # Apply linear transformation
        attention_scores = F.softmax(self.attention_weights(pooled_data), dim=1)  # Calculate attention scores
        pooled_data = torch.matmul(attention_scores.transpose(2, 3), pooled_data)  # Apply attention pooling
        return pooled_data

class DualTemporalGraphLearning(nn.Module):
    def __init__(self, srl_input_size, srl_hidden_size, gcn_input_size, gcn_hidden_size, lstm_hidden_size, num_classes):
        super(DualTemporalGraphLearning, self).__init__()
        self.srl = SignalRepresentationLearning(srl_input_size, srl_hidden_size)
        self.gcn = TemporalGraphRepresentationLearning(gcn_input_size, gcn_hidden_size, lstm_hidden_size, num_classes)

    def forward(self, raw_bold_signals, coarsened_graphs):
        srl_output = self.srl(raw_bold_signals)
        gcn_output = self.gcn(coarsened_graphs, srl_output)
        return gcn_output

class SignalRepresentationLearning(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SignalRepresentationLearning, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, input_data):
        conv_output = F.relu(self.conv(input_data.transpose(1, 2)))  # Apply 1D convolution
        return conv_output.transpose(1, 2)  # Transpose back to (batch_size, seq_len, hidden_size)

class TemporalGraphRepresentationLearning(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_hidden_size, num_classes):
        super(TemporalGraphRepresentationLearning, self).__init__()
        self.gcn = GraphConvolution(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, input_graphs, srl_embeddings):
        # Apply graph convolution
        gcn_output = self.gcn(input_graphs)

        # LSTM input shape: (batch_size, sequence_length, hidden_size)
        lstm_input = gcn_output.unsqueeze(1)  # Add time dimension
        lstm_output, _ = self.lstm(lstm_input)

        # Take the last output of LSTM
        lstm_output = lstm_output[:, -1, :]

        # Final classification layer
        output = self.fc(lstm_output)
        return output

class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_graph):
        # Apply linear transformation
        output_graph = self.linear(input_graph)
        return output_graph
