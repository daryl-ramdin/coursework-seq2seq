import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUEncoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_vocab, sizeof_embedding):
        super(GRUEncoder, self).__init__()

        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding

        # Create and embedding
        self.embedding = nn.Embedding(self.sizeof_vocab, self.sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding, batch_first=True)

    def forward(self, input_tensor):
        output_tensor = self.embedding(input_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor)
        return output_tensor, hidden_tensor


class GRUDecoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_embedding, sizeof_vocab, **kwargs):
        super(GRUDecoder, self).__init__()

        self.sizeof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.embedding = nn.Embedding(sizeof_vocab, sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding, batch_first=True)
        self.out = nn.Linear(self.sizeof_embedding, self.sizeof_vocab)

    def forward(self, input_tensor, hidden_tensor):
        output_tensor = self.embedding(input_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor, hidden_tensor)
        output_tensor = self.out(output_tensor)
        output_tensor = F.log_softmax(output_tensor,dim=2)
        return output_tensor, hidden_tensor


class AttentionDecoder(nn.Module):

    def __init__(self, sizeof_embedding, sizeof_vocab, seq_length):
        super(AttentionDecoder, self).__init__()

        self.sizof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab
        self.seq_length = seq_length

        self.embedding = nn.Embedding(sizeof_vocab,sizeof_embedding)

        self.attn = nn.Linear(sizeof_embedding*2,seq_length)

        self.attn_combine = nn.Linear(sizeof_embedding*2, sizeof_embedding)

        self.gru = nn.GRU(input_size=self.sizof_embedding, hidden_size=sizeof_embedding, batch_first=True)

        self.out = nn.Linear(self.sizof_embedding, self.sizeof_vocab)

    def forward(self,input_tensor, hidden_tensor,encoder_output):
        '''
        :param input_tensor: batch x seq_length
        :param hidden_tensor: 1 x batch x sizeof_embedding
        :param encoder_output: batch x seq_length
        :return:
        '''
        #input tensor (batch size , 1)
        input_embedding = self.embedding(input_tensor) # batch_size, 1, 128

        #combined_tensor (batch_size, sizeof_hidden * 2)
        combined_tensor = torch.cat( (input_embedding.squeeze(1),hidden_tensor[0]), 1)

        #attn_weights (batch_size,seq_len)
        attn_weights = self.attn(combined_tensor)

        #Get the probability of each token
        attn_weights = F.softmax(attn_weights,dim=1)

        #Apply the attention weights to the encoder_output
        attn_weights = attn_weights.unsqueeze(1)
        #attn_weights (1, batch_size, seq_len)
        #encoder_output (1, batch_size, seq_len)

        attn_applied = torch.bmm(attn_weights,encoder_output)

        #Combine the attention with the decoder input
        output_tensor = torch.cat((attn_applied[:,0], input_embedding[:,0]),1)

        output_tensor = self.attn_combine(output_tensor)

        output_tensor, hidden_tensor = self.gru(output_tensor.unsqueeze(1), hidden_tensor)

        output_tensor = self.out(output_tensor)

        output_tensor =  F.log_softmax(output_tensor,dim=2)

        return output_tensor, hidden_tensor, attn_weights




