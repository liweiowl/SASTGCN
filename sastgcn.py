import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configuration import cf


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc11 = nn.Linear(cf.seq_len * cf.num_nodes, cf.a_hidden_dim1)
        self.fc12 = nn.Linear(cf.a_hidden_dim1, cf.a_hidden_dim2)
        self.fc21 = nn.Linear(cf.a_hidden_dim2, cf.a_hidden_dim1)
        self.fc22 = nn.Linear(cf.a_hidden_dim1, cf.seq_len * cf.num_nodes)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e2 = F.relu(self.fc12(e1))
        return e2

    def decode(self, x):
        d1 = F.relu(self.fc21(x))
        d2 = F.relu(self.fc22(d1))
        return d2

    def forward(self, x):
        x = x.view(-1, cf.seq_len * cf.num_nodes)
        hidden_repre = self.encode(x)
        autooutput = self.decode(hidden_repre)
        autooutput = autooutput.view(-1, cf.seq_len, cf.num_nodes)
        if cf.return_hidden:
            return autooutput, hidden_repre
        return autooutput


class MultiVariateLSTM(nn.Module):
    def __init__(self, input_size=cf.gcn_hidden_dim2 * 2, hidden_size=cf.lstm_hidden_size,
                 num_layers=cf.lstm_num_layers, output_size=cf.lstm_output_size,
                 batch_size=cf.batch_size):
        super().__init__()
        self.input_size = input_size  # node number, number of variable
        self.hidden_size = hidden_size  # number of node in hidden layer, set at will
        self.num_layers = num_layers  # how many lstm layer are stacked together
        self.output_size = output_size  #
        self.batch_size = batch_size
        self.num_directions = 1  # =2 if bidirectional else 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(cf.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(cf.device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]  # here is cf.seq_len   30
        # input (batch_size, seq_len, input_size)  (32, 30, 207)
        input_seq = input_seq.view(self.batch_size, seq_len,
                                   self.input_size)  # do not need sometimes if the input_seq shape is correct
        # output  (batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # here _ represents (h_n, c_n)
        # print("output.size=", output.size())
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (32*30, 200)
        # print("output.size 2=", output.size())
        pred = self.fc(output)
        # print("pred shape = ", pred.shape)  #(960,207)
        pred = pred.view(self.batch_size, seq_len, -1)

        pred = pred[:, -cf.pred_len:,
               :]  # the final lstm output contains predictions of all time slice, we only need the later part
        # print("pred shape 2 = ", pred.shape)  #(32,6,207)
        return pred


class AutoEncoder_LSTM(nn.Module):
    def __init__(self):
        super(AutoEncoder_LSTM, self).__init__()
        self.lstmblock = MultiVariateLSTM()
        self.selfsupervision = AutoEncoder()
        self.fc = nn.Linear(in_features=60 + cf.num_nodes * cf.pred_len, out_features=207 * 6)

    def forward(self, x):
        autooutput, hidden_fea = self.selfsupervision(x)
        bias = x - autooutput
        # x and bias shape is (batch_size, seq_len, num_nodes), hidden_fea shape is (batch_size, 60)
        # print(f"x.shape={x.shape}, bias.shape={bias.shape}, hidden_fea.shape={hidden_fea.shape}")
        new_x = torch.cat((x, bias), 0)
        # new_x shape is (2*batch_size, seq_len, num_nodes)
        # print(f"new_x.shape={new_x.shape}")
        # lstm_out shape is  (2*batch_size, seq_len, num_nodes)
        lstm_out = self.lstmblock(new_x)
        # print(f"lstm_out.shape={lstm_out.shape}")
        lstm_out = lstm_out.view(cf.batch_size, -1)
        new_feature = torch.cat((lstm_out, hidden_fea), 1)
        # print(f"new_feature.shape={new_feature.shape}")
        out = self.fc(new_feature)
        # print(f"out.shape={out.shape}")
        out = out.view(-1, cf.pred_len, cf.num_nodes)

        return out, autooutput


class SASTGCN(nn.Module):
    def __init__(self):
        super(SASTGCN, self).__init__()
        self.stblock = STBlock()
        self.selfsupervision = AutoEncoder()
        self.fc = nn.Linear(in_features=cf.a_hidden_repre_dim+cf.lstm_output_size * cf.pred_len, out_features=cf.num_nodes * cf.pred_len)
        # self.adj = adj

    def forward(self, x, adj):
        autooutput, hidden_fea = self.selfsupervision(x)
        bias = x - autooutput
        new_x = torch.cat((x, bias), 0)
        st_out = self.stblock(new_x, adj)
        # print(f"st_out.shape={st_out.shape}")
        st_out = st_out.view(cf.batch_size, -1)
        new_feature = torch.cat((st_out, hidden_fea), 1)
        # print(f"new_feature.shape={new_feature.shape}")
        out = self.fc(new_feature)
        # print(f"out.shape={out.shape}")
        out = out.view(-1, cf.pred_len, cf.num_nodes)

        return out, autooutput


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        # print(f"adj shape={adj.shape}, features shape={features.shape}")
        out = torch.mm(adj, features)  # graph = f(A_hat*W)
        # print(f'out.shape={out.shape}')
        # out = torch.transpose(out,1,0)
        out = self.linear(out.transpose(1, 0))
        return out


class GCN(torch.nn.Module):
    def __init__(self, input_size=cf.num_nodes, hidden_size=cf.gcn_hidden_dim1, output_size=cf.gcn_hidden_dim2):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, output_size)

    def forward(self, adj, features):
        # print(f"adj shape {adj.shape}, features shape {features.shape}")
        out = torch.nn.functional.relu(self.gcn1(adj, features))
        # print(f"out shape = {out.shape}")
        out = self.gcn2(adj, out.transpose(1, 0))
        return out


class STBlock(nn.Module):
    def __init__(self):
        super(STBlock, self).__init__()
        self.st_gcn = GCN()
        self.st_lstm = MultiVariateLSTM()

    def forward(self, input_feature, adj):
        lstm_inputs = []
        for i in range(cf.seq_len):
            # print(f"adj.shape={adj.shape}")
            # print(f"input feature.shape ={input_feature.shape}")
            out1 = self.st_gcn(adj, input_feature[:, i, :].transpose(1, 0))
            # print(f"temp out shape={out1.shape}")
            if i == 0:
                lstm_inputs = out1
            else:
                lstm_inputs = torch.vstack((lstm_inputs, out1))
        # lstm_inputs = torch.tensor(np.array(lstm_inputs)).to(cf.device)
        # print(f"lstm_input_shape={lstm_inputs.shape}")
        lstm_inputs = lstm_inputs.view(cf.batch_size, cf.seq_len, -1)
        # print(f"lstm_input_shape={lstm_inputs.shape}")
        lstm_outputs = self.st_lstm(lstm_inputs)
        return lstm_outputs
