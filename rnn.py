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

    def forward(self, input_tensor, hidden_tensor):
        output_tensor = self.embedding(input_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor, hidden_tensor)
        return output_tensor, hidden_tensor


class GRUDecoder(nn.Module):
    # ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self, sizeof_embedding, sizeof_vocab):
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

        self.embedding = nn.Embedding(sizeof_vocab,sizeof_embedding)

        self.attn = nn.Linear(sizeof_embedding*2,seq_length)

    def forward(self,input_tensor, hidden_tensor,encoder_output):

        output_tensor = self.embedding(input_tensor)
        attn_weights = self.attn(torch.cat(output_tensor,hidden_tensor))
        attn_applied = attn_weights*encoder_output


