import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUEncoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_vocab, sizeof_embedding, num_layers=1, bidirectional=False, dropout=0.5):
        super(GRUEncoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding

        # Create and embedding
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.sizeof_vocab, self.sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding,
                          num_layers=self.num_layers, bidirectional=self.bidirectional)

    def forward(self, input_tensor, sequence_lengths):
        '''
        :param input_tensor: (sequence_length, batch_size)
        :return: output_tensor, hidden_tensor. Shapes are:
                 output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
                 hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        '''

        # Get the embedding for the input tensor
        embedded_tensor = self.embedding(input_tensor)  # (sequence_length, batch_size, sizeof_embedding)

        embedded_tensor = self.dropout(embedded_tensor)

        # Pack padded batch of sequences for RNN module
        # embedded_tensor = nn.utils.rnn.pack_padded_sequence(embedded_tensor, sequence_lengths, enforce_sorted=False)

        # Encode the embedded tensor
        output_tensor, hidden_tensor = self.gru(embedded_tensor)

        # Unpack padding
        # output_tensor, _ = nn.utils.rnn.pad_packed_sequence(output_tensor)

        # Return the output and hidden tensors
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
        # hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        return output_tensor, hidden_tensor


class GRUDecoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_embedding, sizeof_vocab, num_layers=1, bidirectional=False, dropout=0.5):
        super(GRUDecoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sizeof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(sizeof_vocab, sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding,
                          num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.out = nn.Linear(self.sizeof_embedding, self.sizeof_vocab)

    def forward(self, input_tensor, hidden_tensor):
        '''
        Decode one token at a time in the batch of sequences
        :param input_tensor:    (1, batch_size)
        :param hidden_tensor:   (D*num_layers in encoder rnn, batch_size, sizeof_embedding)
        :return: output_tensor, hidden_tensor
                Shapes are:
                output_tensor (1,batch_size,sizeof_vocab)
                hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        '''

        # Get the embedding of the input tensor. It will have a shape
        # (1, batch_size, sizeof_embedding)
        embedded_tensor = self.embedding(input_tensor)

        embedded_tensor = self.dropout(embedded_tensor)

        # Decode the embedding tensor and hidden tensor.
        # output_tensor: (1, batch_size, D*sizeof_embedding)
        # hidden_tensor: (D*num_layers, batch_size, sizeof_embedding)
        output_tensor, hidden_tensor = self.gru(embedded_tensor, hidden_tensor)

        # Predict the next token. The shape will then become
        # output_tensor: (1, batch_size, sizeof_vocab)
        output_tensor = self.out(output_tensor)
        # output_tensor = F.softmax(output_tensor, dim=2)
        return output_tensor, hidden_tensor


class AttentionDecoder(nn.Module):

    def __init__(self, sizeof_embedding, sizeof_vocab, seq_length):
        super(AttentionDecoder, self).__init__()

        self.sizof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.seq_length = seq_length

        self.embedding = nn.Embedding(sizeof_vocab, sizeof_embedding)

        self.attn = nn.Linear(sizeof_embedding * 2, seq_length)

        self.attn_combine = nn.Linear(sizeof_embedding * 2, sizeof_embedding)

        self.gru = nn.GRU(input_size=self.sizof_embedding, hidden_size=sizeof_embedding, batch_first=True)

        self.out = nn.Linear(self.sizof_embedding, self.sizeof_vocab)

    def forward(self, input_tensor, hidden_tensor, encoder_output):
        '''
        :param input_tensor: batch x seq_length
        :param hidden_tensor: 1 x batch x sizeof_embedding
        :param encoder_output: batch x seq_length
        :return:
        '''
        # input tensor (batch size , 1)
        input_embedding = self.embedding(input_tensor)  # batch_size, 1, 128

        # combined_tensor (batch_size, sizeof_hidden * 2)
        combined_tensor = torch.cat((input_embedding.squeeze(1), hidden_tensor[0]), 1)

        # attn_weights (batch_size,seq_len)
        attn_weights = self.attn(combined_tensor)

        # Get the probability of each token
        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply the attention weights to the encoder_output
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights (1, batch_size, seq_len)
        # encoder_output (1, batch_size, seq_len)

        attn_applied = torch.bmm(attn_weights, encoder_output)

        # Combine the attention with the decoder input
        output_tensor = torch.cat((attn_applied[:, 0], input_embedding[:, 0]), 1)

        output_tensor = self.attn_combine(output_tensor)

        output_tensor, hidden_tensor = self.gru(output_tensor.unsqueeze(1), hidden_tensor)

        output_tensor = self.out(output_tensor)

        # output_tensor = F.log_softmax(output_tensor, dim=2)

        return output_tensor, hidden_tensor, attn_weights


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size parameters are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
