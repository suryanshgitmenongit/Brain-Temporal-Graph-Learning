import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DynamicFCDataset(Dataset):
    def __init__(self, data, labels, window_length, stride, sub_seq_length):
        self.data = data
        self.labels = labels
        self.window_length = window_length
        self.stride = stride
        self.sub_seq_length = sub_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series = self.data[idx]
        # Crop the time series to a fixed length
        time_series = time_series[:self.sub_seq_length]
        dynamic_fc, bold_signals = self.compute_dynamic_fc_and_bold(time_series)
        return torch.FloatTensor(dynamic_fc), torch.FloatTensor(bold_signals), torch.LongTensor([self.labels[idx]])

    def compute_dynamic_fc_and_bold(self, time_series):
        num_windows = (time_series.shape[0] - self.window_length) // self.stride + 1
        dynamic_fc = np.zeros((num_windows, time_series.shape[1], time_series.shape[1]))
        bold_signals = np.zeros((num_windows, time_series.shape[1]))
        
        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_length
            window = time_series[start:end, :]
            fc = np.corrcoef(window.T)
            dynamic_fc[i] = fc
            bold_signals[i] = window.mean(axis=0)
        
        return dynamic_fc, bold_signals

def load_and_preprocess_data(A, B, batch_size, window_length, stride, sub_seq_length, num_sub_seqs):
    # Combine A and B
    data = np.concatenate((A, B), axis=0)
    
    # Create labels
    labels = np.concatenate((np.zeros(len(A)), np.ones(len(B))))
    
    # Split data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DynamicFCDataset(train_data, train_labels, window_length, stride, sub_seq_length)
    test_dataset = DynamicFCDataset(test_data, test_labels, window_length, stride, sub_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

class SignalRepresentationLearning(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(SignalRepresentationLearning, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        return x.permute(0, 2, 1)

class AttentionGraphPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionGraphPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.F = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.F)

    def forward(self, x):
        batch_size, num_windows, num_regions, _ = x.size()
        S = F.softmax(torch.matmul(x.view(-1, num_regions, num_regions), self.F), dim=1)
        S = S.view(batch_size, num_windows, num_regions, self.output_dim)
        A_hat = torch.matmul(S.transpose(2, 3), torch.matmul(x, S))
        return A_hat

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output

class DTGL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gcn_layers, num_lstm_layers, max_skip):
        super(DTGL, self).__init__()
        self.gcn_layers = nn.ModuleList([GCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) 
                                         for i in range(num_gcn_layers)])
        self.lstm_layers = nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim, batch_first=True) 
                                          for _ in range(max_skip)])
        self.max_skip = max_skip

    def forward(self, x, adj):
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, adj))
        
        outputs = []
        for p in range(1, self.max_skip + 1):
            h = x
            for t in range(p-1, x.size(1), p):
                h_t, _ = self.lstm_layers[p-1](h[:, t-p+1:t+1, :])
                h[:, t, :] = h_t[:, -1, :]
            outputs.append(h)
        
        combined_output = torch.stack(outputs, dim=-1).mean(dim=-1)
        return combined_output

class BrainTGL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, num_lstm_layers, max_skip):
        super(BrainTGL, self).__init__()
        self.srl = SignalRepresentationLearning(input_dim, hidden_dim)
        self.graph_pooling = AttentionGraphPooling(input_dim, hidden_dim)
        self.dtgl = DTGL(hidden_dim, hidden_dim, num_gcn_layers, num_lstm_layers, max_skip)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.skip_connection1 = nn.Linear(input_dim, hidden_dim)
        self.skip_connection2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, bold_signals):
        srl_output = self.srl(bold_signals)
        pooled = self.graph_pooling(x)
        skip1 = self.skip_connection1(x.mean(dim=1).mean(dim=1))
        dtgl_output = self.dtgl(pooled + skip1.unsqueeze(1) + srl_output, x.mean(dim=1))
        skip2 = self.skip_connection2(pooled.mean(dim=1))
        output = self.output_layer(dtgl_output[:, -1, :] + skip2)
        return output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, bold, target) in enumerate(train_loader):
        data, bold, target = data.to(device), bold.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, bold)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device, num_sub_seqs):
    model.eval()
    total_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, bold, target in test_loader:
            data, bold, target = data.to(device), bold.to(device), target.to(device)
            output = model(data, bold)
            total_loss += criterion(output, target.squeeze()).item()
            pred = output.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Voting scheme for subject-level prediction
    subject_predictions = []
    subject_targets = []
    for i in range(0, len(all_predictions), num_sub_seqs):
        subject_pred = np.bincount(all_predictions[i:i+num_sub_seqs].flatten()).argmax()
        subject_predictions.append(subject_pred)
        subject_targets.append(all_targets[i])
    
    correct = np.sum(np.array(subject_predictions) == np.array(subject_targets))
    accuracy = correct / len(subject_targets)
    
    return total_loss / len(test_loader), accuracy

def main():
    # Load your data
    A = np.load('path_to_A.npy')  # Shape: (num_subjects, num_timepoints, num_regions)
    B = np.load('path_to_B.npy')  # Shape: (num_subjects, num_timepoints, num_regions)

    # Hyperparameters
    input_dim = A.shape[2]  # Number of brain regions
    hidden_dim = 64
    output_dim = 2  # Binary classification
    num_gcn_layers = 2
    num_lstm_layers = 1
    max_skip = 3
    window_length = 20
    stride = 5
    sub_seq_length = 200  # Length of sub-sequences
    num_sub_seqs = 5  # Number of sub-sequences per subject
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = BrainTGL(input_dim, hidden_dim, output_dim, num_gcn_layers, num_lstm_layers, max_skip).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load and preprocess data
    train_loader, test_loader = load_and_preprocess_data(A, B, batch_size, window_length, stride, sub_seq_length, num_sub_seqs)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, num_sub_seqs)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
