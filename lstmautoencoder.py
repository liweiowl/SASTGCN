import numpy as np
import torch.nn as nn 


class RecurrentEncoder(nn.Module):
	def __init__(self, seq_len, n_features, embedding_dim=64):
		super(RecurrentEncoder, self).__init__()

		self.seq_len, self.n_features = seq_len, n_features
		self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

		self.rnn1 = nn.LSTM(
			input_size=n_features,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_size=True
			)

		self.rnn2 = nn.LSTM(
			input_size=self.hidden_dim,
			hidden_size=embedding_dim,
			num_layers=1,
			batch_first=True)

	def forward(self, x):
		x = x.reshape((1, self.seq_len, self.n_features))
		x, (_, _) = self.rnn1(x)
		x, (hidden_n, _) = self.rnn2(x)
		x = hidden_n.reshape((self.n_features, self.embedding_dim))
		return x

class RecurrentDecoder(nn.Module):
	def __init__(self, seq_len, input_dim=64, n_features=1):
		super(RecurrentDecoder, self).__init__()

		self.seq_len, self.input_dim = seq_len, input_dim
		self.hidden_dim, self.n_features = 2 * input_dim, n_features

		self.rnn3 = nn.LSTM(
			input_size=input_dim,
			hidden_size=input_dim,
			num_layers=1,
			batch_first=True)

		self.rnn4 = nn.LSTM(
			input_size=input_dim,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_first=True)

		self.output_layer = nn.Linear(self.hidden_dim, n_features)

	def forward(self, x):
		x = x.repeat(self.seq_len, self.n_features)
		x = x.reshape((self.n_features, self.seq_len, self.input_dim))
		x, (hidden_n, cell_n) = self.rnn3(x)
		x, (hidden_n, cell_n) = self.rnn4(x)
		x = x.reshape((self.seq_len, self.hidden_dim))
		return self.output_layer(x)


class RecurrentAutoEncoder(nn.Module):
	def __init__(self, seq_len, n_features, embedding_dim):
		super(RecurrentAutoEncoder, self).__init__()
		self.rnnencoder = RecurrentEncoder(seq_len, n_features, embedding_dim).to(device)
		self.rnndecoder = RecurrentDecoder(seq_len, embedding_dim, n_features).to(device)

	def forward(self, x):
		temporal_feature = self.rnnencoder(x)
		x = self.rnndecoder(temporal_feature)
		return x, temporal_feature


