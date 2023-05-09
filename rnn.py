import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUEncoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_vocab, sizeof_embedding, num_layers=1, bidirectional=False, dropout=0):
        super(GRUEncoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding
        self.dropout = dropout

        # Create and embedding
        self.embedding = nn.Embedding(self.sizeof_vocab, self.sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding,
                          num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout)

    def forward(self, input_tensor, sequence_lengths):
        '''
        :param input_tensor: (sequence_length, batch_size)
        :return: output_tensor, hidden_tensor. Shapes are:
                 output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
                 hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        '''

        # Get the embedding for the input tensor
        embedded_tensor = self.embedding(input_tensor)  # (sequence_length, batch_size, sizeof_embedding)

        # Pack padded batch of sequences for RNN module
        embedded_tensor = nn.utils.rnn.pack_padded_sequence(embedded_tensor, sequence_lengths, enforce_sorted=False)

        # Encode the embedded tensor
        output_tensor, hidden_tensor = self.gru(embedded_tensor)

        # Unpack padding
        output_tensor, _ = nn.utils.rnn.pad_packed_sequence(output_tensor)

        # If bidrectional, sum the outputs outputs ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        if self.bidirectional:
            output_tensor = output_tensor[:, :, :self.sizeof_embedding] + output_tensor[:, :, self.sizeof_embedding:]

        # Return the output and hidden tensors
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
        # hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        return output_tensor, hidden_tensor


class GRUDecoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_embedding, sizeof_vocab, num_layers=1, dropout=0):
        super(GRUDecoder, self).__init__()

        self.num_layers = num_layers
        self.sizeof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.embedding = nn.Embedding(sizeof_vocab, sizeof_embedding)
        self.dropout = dropout

        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding,
                          num_layers=self.num_layers, dropout=self.dropout)
        self.out = nn.Linear(self.sizeof_embedding, self.sizeof_vocab)

    def forward(self, input_tensor, hidden_tensor, encoder_output=None):
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

        # Decode the embedding tensor and hidden tensor.
        # output_tensor: (1, batch_size, D*sizeof_embedding)
        # hidden_tensor: (D*num_layers, batch_size, sizeof_embedding)
        output_tensor, hidden_tensor = self.gru(embedded_tensor, hidden_tensor)

        # Predict the next token. The shape will then become
        # output_tensor: (1, batch_size, sizeof_vocab)
        output_tensor = self.out(output_tensor)

        #Remove the first dimension
        output_tensor = output_tensor.squeeze(0)
        return output_tensor, hidden_tensor


class AttentionEncoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_vocab, sizeof_embedding, num_layers=1, bidirectional=True, dropout=0):
        super(AttentionEncoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding
        self.dropout = dropout

        # Create and embedding
        self.embedding = nn.Embedding(self.sizeof_vocab, self.sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding,
                          num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout)

    def forward(self, input_tensor, sequence_lengths):
        '''
        :param input_tensor: (sequence_length, batch_size)
        :return: output_tensor, hidden_tensor. Shapes are:
                 output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
                 hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        '''

        # Get the embedding for the input tensor
        embedded_tensor = self.embedding(input_tensor)  # (sequence_length, batch_size, sizeof_embedding)

        # Pack padded batch of sequences for RNN module ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        embedded_tensor = nn.utils.rnn.pack_padded_sequence(embedded_tensor, sequence_lengths, enforce_sorted=False)

        # Encode the embedded tensor
        output_tensor, hidden_tensor = self.gru(embedded_tensor)

        # Unpack padding ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        output_tensor, _ = nn.utils.rnn.pad_packed_sequence(output_tensor)

        # If bidrectional, sum the outputs ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        if self.bidirectional:
            output_tensor = output_tensor[:, :, :self.sizeof_embedding] + output_tensor[:, :, self.sizeof_embedding:]

        # Return the output and hidden tensors
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # output_tensor (sequence_length, batch_size, D * sizeof_embedding) where D = 1 or 2 if bi-directional
        # hidden_tensor (D*num_layers, batch_size, sizeof_embedding)
        return output_tensor, hidden_tensor


class Attention(nn.Module):
    def __init__(self, sizeof_vocab, sizeof_embedding):
        super(Attention, self).__init__()

        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding

        # ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        self.weight_matrices = torch.nn.Linear(sizeof_embedding * 2, sizeof_embedding)
        self.weight_vector = nn.Parameter(torch.FloatTensor(sizeof_embedding))

    def forward(self, hidden_tensor, annotation):
        # ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        attn_score = self.weight_matrices(
            torch.cat((hidden_tensor.expand(annotation.size(0), -1, -1), annotation), 2)).tanh()
        attn_score = torch.sum(self.weight_vector * attn_score, dim=2)

        # Transpose the score
        attn_score = attn_score.t()

        # Return the weight as the softmax of the scores
        return F.softmax(attn_score, dim=1).unsqueeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, sizeof_embedding, sizeof_vocab, num_layers, dropout=0):
        super(AttentionDecoder, self).__init__()

        # ref https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        self.sizeof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.dropout = dropout
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.sizeof_vocab, self.sizeof_embedding)
        self.gru = nn.GRU(self.sizeof_embedding, self.sizeof_embedding, dropout=dropout, num_layers=self.num_layers)
        self.concat = nn.Linear(self.sizeof_embedding * self.num_layers, self.sizeof_embedding)
        self.out = nn.Linear(self.sizeof_embedding, sizeof_vocab)

        self.attention = Attention(sizeof_vocab, self.sizeof_embedding)

    def forward(self, input_tensor, hidden_tensor, encoder_output):
        # ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot%20tutorial
        # Get the embedding for the input and apply dropout if any
        embedded_tensor = self.embedding(input_tensor)

        # GRU
        gru_output, hidden = self.gru(embedded_tensor, hidden_tensor)

        # Calculate attention weights using the attention module
        attn_weights = self.attention(gru_output, encoder_output)

        # Multiply attention weights to encoder outputs to get context vector
        context_vector = attn_weights.bmm(encoder_output.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        gru_output = gru_output.squeeze(0)
        context_vector = context_vector.squeeze(1)
        concat_input = torch.cat((gru_output, context_vector), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        # output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
