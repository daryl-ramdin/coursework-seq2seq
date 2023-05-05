#!/usr/bin/env python
# coding: utf-8

# <h2>Imports</h2>

# In[1]:


import os
import numpy as np
import torch
from corpus import Corpus, CornellMovieCorpus, Vocabulary
from rnn import GRUEncoder, GRUDecoder
from torch import optim
import torch.nn as nn
from matplotlib import pyplot as plt
from datetime import datetime
import random
from torch.utils.data import DataLoader as DataLoader
from metrics import show_loss


# <h2>Parameter Settings</h2>

# In[2]:


random.seed(77)
EXPERIMENT_NAME = "testing"
SIZEOF_EMBEDDING = 128
NUMBER_OF_EPOCHS = 1
PRINT_INTERVAL = 10
BATCH_SIZE = 1
TEACHER_FORCING = 0
TEACHER_FORCING_DECAY = 1e-10
PROGRESS_INTERVAL = 100
LEARNING_RATE = 1e-03
CONVO_MODE = Corpus.FULL


# <h2>Use the GPU if present</h2>

# In[3]:


device = torch.device('cpu')
if (torch.cuda.is_available()):
   device = torch.device('cuda')
print(device)


# <h2>Create a Cornell Movie Corpus </h2>

# In[4]:


corpus = CornellMovieCorpus(convo_mode=CONVO_MODE)


# <h2>Let's look at some data</h2>

# In[5]:


lines = list(corpus.movie_lines.items())
print(len(lines), "movie lines loaded")

distinct_lines = [line[1]["prepped_text"] for line in lines]
print(len(set(distinct_lines)), "distinct movie lines exist")

print(len(corpus.movie_convos), "conversations loaded")                    


# <h2>Create Encoders and Decoders</h2>

# In[6]:


sizeof_vocab = corpus.vocabulary.len

encoder = GRUEncoder(sizeof_vocab, SIZEOF_EMBEDDING)
decoder = GRUDecoder(SIZEOF_EMBEDDING, sizeof_vocab)

encoder.to(device)
decoder.to(device)


# <h2>Let's setup our trainer</h2>

# In[7]:


encoder_optimizer = optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(),lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss()

dataloader = DataLoader(corpus, batch_size=BATCH_SIZE,shuffle=True)

#Let's get a random exchange pair
epoch_loss = 0
start_time = datetime.now()
training_loss = []
total_sequences = 0
interval_sequences = 0

if corpus.convo_mode==corpus.FULL:
    batch_type = "Conversation"
else:
    batch_type = "Exchange Pair"


# <h2>Train</h2>

# In[8]:


for epoch in range(NUMBER_OF_EPOCHS):
    epoch_loss = 0
    processed_total_batches = 0
    batch_counter = 0

    interval_loss = 0
    for idx, batch in enumerate(dataloader):
        
        teacher_forcing = False
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        Q_tensors = batch[1]["Q"] #shape(batch_size,convo_length,seq_len)
        A_tensors= batch[1]["Q"] #shape(batch_size,convo_length,seq_len)
        
        Q_tensors.to(device)
        A_tensors.to(device)

        Q_tensors = Q_tensors.squeeze(0)
        A_tensors = A_tensors.squeeze(0)

        convo_length = A_tensors.shape[0]
        seq_length= A_tensors.shape[1]

        interval_sequences+=len(A_tensors)
        total_sequences+=interval_sequences

        #print("Tensor shapes", Q_tensors.shape, A_tensors.shape)

        #Encode the batch
        encoder_hidden = torch.zeros(1,convo_length,SIZEOF_EMBEDDING,device=device)
        encoder_output, encoder_hidden = encoder(Q_tensors,encoder_hidden) # encoder_output: (batch_size, max_seq_len, hidden_size), encoder_hidden: (1, batch_size, hidden_size)

        # #The decoder accepts an input and the previous hidden start
        # At the start, the first input is the SOS token and the
        # hidden state is the output of the encoder i.e. context vector

        #The first input to the decoder is the SOS token.
        #Iterate through the batch of A_tensors and get the first element in each sequence
        decoder_input = A_tensors[:, 0].view(-1,1) #decoder_input (batch_size, 1)

        #The initial hidden input to the decoder is the last hidden of the encoder.
        #For each sequence in the encoder hidden batch, get the last token. encoder_hidden: (batch_size, max_seq_len, hidden_size)
        # decoder_hidden = encoder_output[:,-1:,:].squeeze(1) 
        # decoder_hidden = decoder_hidden.unsqueeze(0) #decoder_hidden  (1, batch_size, sizeof_hidden)
        decoder_hidden = encoder_hidden
        
        loss = 0
        #If we are using teach forcing then the decoder_hidden is the 
        if random.random() < TEACHER_FORCING: teacher_forcing = True
        
        for i in range(seq_length-1):
            
            #if teacher forcing then the target is fed in as the input
            if teacher_forcing: decoder_input = A_tensors[:,i+1].view(-1,1)
            
            #Get the decoder output and hidden state for the
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) #decoder_output (batch_size, 1, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)

            decoder_output = decoder_output.squeeze(1)
            # calculate the loss by comparing with the tensor of the target
            target = A_tensors[:, i+1]
            loss += criterion(decoder_output, target)

            #Get the top prediction for each batch
            decoder_input = decoder_output.topk(k=1, dim=1).indices

        convo_loss = loss/corpus.max_seq_length
        interval_loss += convo_loss

        
        TEACHER_FORCING = max(0,TEACHER_FORCING - (TEACHER_FORCING*TEACHER_FORCING_DECAY))

        if batch_counter%PROGRESS_INTERVAL == 0 and batch_counter:
            #Calculate the average sequence loss over the interval
            end_time = datetime.now()
            timediff = end_time - start_time
            timediff = timediff.seconds
            print(batch_type + ' Number: {0:1d}, Number of sequences: {1:1d}, Average sequence loss {2:.6f}, in {3:d} seconds'.format(batch_counter,interval_sequences,interval_loss/PROGRESS_INTERVAL,timediff))
            start_time = datetime.now()
            interval_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            interval_loss = 0
            interval_sequences = 0

        batch_counter+=1
        training_loss.append([batch_counter,convo_loss.item()])
        if batch_counter> 200: break


# <h2>Print the results</h2>

# In[ ]:

data_file = EXPERIMENT_NAME + ".csv"
np.savetxt(data_file,training_loss,delimiter=",")
show_loss(data_file,100)




