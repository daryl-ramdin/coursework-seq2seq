#!/usr/bin/env python
# coding: utf-8

# <h2>Imports</h2>

# In[1]:


import os
import numpy as np
import torch
from corpus import Corpus, CornellMovieCorpus, Vocabulary
from rnn import GRUEncoder, GRUDecoder, AttentionDecoder
from torch import optim
import torch.nn as nn
from matplotlib import pyplot as plt
from datetime import datetime
import random
from torch.utils.data import DataLoader as DataLoader
from metrics import show_loss
from utils import  evaluate


# <h2>Parameter Settings</h2>

# In[2]:


random.seed(77)
EXPERIMENT_NAME = "exp3"
SIZEOF_EMBEDDING = 128
NUMBER_OF_EPOCHS = 1
BATCH_SIZE = 5
TEACHER_FORCING = 0.5
TEACHER_FORCING_DECAY = 0
PROGRESS_INTERVAL = 1000
EVAL_INTERVAL = 1000
LEARNING_RATE = 1e-03
CONVO_MODE = Corpus.PAIRS


# <h2>Use the GPU if present</h2>

# In[3]:


device = torch.device('cpu')
if (torch.cuda.is_available()):
   device = torch.device('cuda')
print(device)


# <h2>Create a Cornell Movie Corpus </h2>

# In[4]:


corpus = CornellMovieCorpus(device=device, data_directory = "data",limit_pairs=True, limit_words=True)


# <h2>Let's look at some data</h2>

# In[5]:


lines = list(corpus.movie_lines.items())
print(len(lines), "movie lines loaded")

distinct_lines = [line[1]["prepped_text"] for line in lines]
print(len(set(distinct_lines)), "distinct movie lines exist")

line_lengths = [len(line[1]["prepped_text"].split(" ")) for line in lines]
print(sum(line_lengths)/len(line_lengths), "Average number of tokens in a line")


print(len(corpus.movie_convos), "conversations loaded")

print(len(corpus.conversations), "exchange pairs")


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

#ref: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3?u=ptrblck
dataloader = DataLoader(corpus, batch_size=BATCH_SIZE,shuffle=True, collate_fn=Corpus.collate_convos)


batch_type = "Exchange Pair"


# <h2>Train</h2>

# In[15]:


epoch_loss = 0
start_time = datetime.now()
training_loss = []
total_sequences = 0
interval_seq_count = 0

for epoch in range(NUMBER_OF_EPOCHS):
    epoch_loss = 0
    processed_total_batches = 0
    batch_counter = 0

    interval_loss = 0
    for idx, batch in enumerate(dataloader):
        #The batch contains a set of exchange pairs
        
        teacher_forcing = False
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        Q_tensors = torch.stack([batch[i][1]["Q"] for i in range(len(batch))]) #shape(batch_size, 1, seq_len) #1 represents 1 exchange pair
        A_tensors= torch.stack([batch[i][1]["A"] for i in range(len(batch))]) #shape(batch_size, 1, seq_len)  #1 represents 1 exchange pair

        Q_lens = [len(batch[i][1]["conversation"]["Q"]["indices"]) for i in range(len(batch))]
        A_lens = [len(batch[i][1]["conversation"]["A"]["indices"]) for i in range(len(batch))]

        Q_tensors = Q_tensors.squeeze(1)
        A_tensors = A_tensors.squeeze(1)

        #Get the batch size and sequence length
        batch_size = A_tensors.shape[0]
        seq_length = A_tensors.shape[1]

        #Transpose to tensors are now seql_length, batch_size
        Q_tensors = torch.transpose(Q_tensors,0,1)
        A_tensors = torch.transpose(A_tensors,0,1)
        
        encoder_outputs = torch.zeros(seq_length, encoder.sizeof_embedding, device=device)

        for ei in range(seq_length):
            cur = Q_tensors[ei,:].view(1,-1)
            encoder_output, encoder_hidden = encoder(cur)
            encoder_outputs[ei] = encoder_output[0, 0]

        # At the start, the first input to the decoder is the SOS token
        # and last hidden state of the encoder

        #To get the SOS token, iterate through the batch of A_tensors
        # and get the first element in each sequence
        decoder_input = A_tensors[0,:].view(1,-1) #decoder_input (1,batch_size)

        #The initial hidden input to the decoder is the last hidden of the encoder.
        #For each sequence in the encoder hidden batch, get the last token. encoder_hidden: (batch_size, max_seq_len, hidden_size)
        # decoder_hidden = encoder_output[:,-1:,:].squeeze(1) 
        # decoder_hidden = decoder_hidden.unsqueeze(0) #decoder_hidden  (1, batch_size, sizeof_hidden)
        decoder_hidden = encoder_hidden
        
        loss = 0
        interval_seq_count+=batch_size
        total_sequences+=batch_size

        #If we are using teach forcing then the decoder_hidden is the 
        if random.random() < TEACHER_FORCING: teacher_forcing = True
        
        for i in range(1, seq_length):
            
            #Get the decoder output and hidden state for the
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) #decoder_output (1, batch_size, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)

            decoder_output = decoder_output.squeeze(0)
            # calculate the loss by comparing with the tensor of the target
            target = A_tensors[i,:]
            loss += criterion(decoder_output, target)

            #Get the top prediction for each batch
            decoder_input = decoder_output.topk(k=1, dim=1).indices

            #Change the view to (1, batch_size)
            decoder_input = decoder_input.view(1,-1)
            
            #if teacher forcing then the target is fed in as the input
            #change view to 1, batch_size
            if teacher_forcing: decoder_input = A_tensors[i,:].view(1,-1)

        sequence_loss = loss.item()/corpus.max_seq_length
        interval_loss += sequence_loss
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        
        TEACHER_FORCING = max(0,TEACHER_FORCING - (TEACHER_FORCING*TEACHER_FORCING_DECAY))

        if batch_counter%PROGRESS_INTERVAL == 0 and batch_counter:
            #Calculate the average sequence loss over the interval
            end_time = datetime.now()
            timediff = end_time - start_time
            timediff = timediff.seconds
            print("Batch #: {0:1d}, Total Sequences Processed: {1:1d}, Interval Size (sequences): {2:1d}, Average sequence loss {3:.6f}, Duration {4:d} seconds, with Teacher Forcing {5:.6f}".format(batch_counter, total_sequences, interval_seq_count, interval_loss/interval_seq_count, timediff, TEACHER_FORCING))
            start_time = datetime.now()
            interval_loss = 0
            interval_seq_count = 0

        if batch_counter%EVAL_INTERVAL == 0:
            evaluate("What is your favourite food?",encoder,decoder,corpus, device)

        batch_counter+=1
        training_loss.append([batch_counter,sequence_loss])


# <h2>Save and print results</h2>

# In[16]:


data_file = EXPERIMENT_NAME + ".csv"
np.savetxt(data_file,training_loss,delimiter=",")


torch.save(encoder.state_dict(), EXPERIMENT_NAME + "_encoder.dict")
torch.save(decoder.state_dict(), EXPERIMENT_NAME + "_decoder.dict")


# In[17]:


show_loss(data_file,1000)


# In[ ]:




