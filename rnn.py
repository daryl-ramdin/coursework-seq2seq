import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    #ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self,sizeof_vocab, sizeof_embedding):
        super(Encoder, self).__init__()

        self.sizeof_vocab = sizeof_vocab
        self.sizeof_embedding = sizeof_embedding

        #Create and embedding
        self.embedding = nn.Embedding(self.sizeof_vocab,self.sizeof_embedding)
        self.gru = nn.GRU(input_size = self.sizeof_embedding, hidden_size = self.sizeof_embedding, batch_first=True)

    def forward(self,input_tensor,hidden_tensor):
        output_tensor = self.embedding(input_tensor).view(1,1,-1) #Change the shape to match the input expected by the GRU
        output_tensor, hidden_tensor = self.gru(output_tensor,hidden_tensor)
        return output_tensor, hidden_tensor

class Decoder(nn.Module):
    #ref for this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self,sizeof_embedding,sizeof_vocab):
        super(Decoder, self).__init__()

        self.sizeof_embedding = sizeof_embedding
        self.sizeof_vocab = sizeof_vocab

        self.embedding = nn.Embedding(sizeof_vocab,sizeof_embedding)
        self.gru = nn.GRU(input_size=self.sizeof_embedding, hidden_size=self.sizeof_embedding, batch_first=True)
        self.out = nn.Linear(self.sizeof_embedding,self.sizeof_vocab)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        output_tensor = self.embedding(input_tensor).view(1, 1, -1)
        output_tensor = F.relu(output_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor, hidden_tensor)
        output_tensor = self.out(output_tensor[0])
        output_tensor = self.softmax(output_tensor)
        return output_tensor, hidden_tensor