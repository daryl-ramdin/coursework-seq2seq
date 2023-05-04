#!/usr/bin/env python
# coding: utf-8

# <h2>Imports</h2>

# In[1]:


import os
import numpy as np
import torch
from corpus import CornellMovieCorpus, Vocabulary
from rnn import Encoder, Decoder
from torch import optim
import torch.nn as nn
from matplotlib import pyplot as plt
from datetime import datetime
import random


# <h2>Use the GPU if present</h2>

# In[2]:


#Let's do the GPU stuff
device = torch.device('mps')
if (torch.cuda.is_available()):
   device = torch.device('cuda')
print(device)


# <h2>Create a Cornell Movie Corpus </h2>

# In[3]:


random.seed(77)

corpus = CornellMovieCorpus()


# <h2>Let's look at some data</h2>

# In[4]:


lines = list(corpus.movie_lines.items())
print(len(lines), "movie lines loaded")

distinct_lines = [line[1]["prepped_text"] for line in lines]
print(len(set(distinct_lines)), "distinct movie lines exist")

print(len(corpus.movie_convos), "conversations loaded")


print("\nExchanges\n")
print(len(corpus.exchange_pairs), "exchanges created")

for i in range(5):
    print("\n",corpus.exchange_pairs[i])

distinct_exchange_pairs = [pair["Q"]["tokens"] + " " + pair["A"]["tokens"] for pair in corpus.exchange_pairs]
print(len(set(distinct_exchange_pairs)), "distinct exchanges exist")                           


# <h2>Create the vocabulary</h2>

# In[5]:


#Let's get a batch of exchanges
pairs, batch = corpus.getBatchExchangeTensors(1)

# for i in range(len(pairs)):
#     print(pairs[i]["Q"],"\n")
#     print(batch[0][i],"\n")
#     print(pairs[i]["A"],"\n")
#     print(batch[1][i],"\n")


# <h2>Create Encoders and Decoders</h2>

# In[6]:


sizeof_embedding = 256
sizeof_vocab = corpus.vocabulary.len

encoder = Encoder(sizeof_vocab, sizeof_embedding)
decoder = Decoder(sizeof_embedding, sizeof_vocab)


# <h2>Let's setup our trainer</h2>

# In[7]:


number_of_epochs = 5000
print_interval = 10
batch_size = 1
teacher_forcing = 0
teacher_forcing_decay = 0

# encoder_optimizer = optim.Adam(encoder.parameters(),lr=1e-03)
# decoder_optimizer = optim.Adam(decoder.parameters(),lr=1e-03)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-03, weight_decay=1e-03)

criterion = nn.CrossEntropyLoss()

#Let's get a random exchange pair
epoch_loss = 0
start_time = datetime.now()
process_before_update = 100
current_batch = 0
total_phrase_pairs = 0
loss_batch = 0
total_epoch_loss = []
for epoch in range(number_of_epochs):
    epoch_loss = 0
    processed_total_batches = 0
    batch_count = 0

    for i in range(75000):

        if not current_batch:
           total_phrase_pairs = 0
           loss_batch = 0
        optimizer.zero_grad()
        current_batch += 1

        pairs, batch = corpus.getBatchExchangeTensors(batch_size)

        Q_tensors = batch[0] #shape(batch_size,max_seq_len)
        A_tensors= batch[1] #shape(batch_size,max_seq_len)
        #print("Tensor shapes", Q_tensors.shape, A_tensors.shape)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        #optimizer.zero_grad()

        #input_tensor, target_tensor = corpus.pairToTensor(exchange_pair)
        #print("Input tensor shape", input_tensor.shape, "target", target_tensor.shape)

        #Encode the batch
        output, hidden = encoder(Q_tensors)# output:(batch_size, max_seq_len, hidden_size), hidden: (1, batch_size, hidden_size)

        # #The decoder accepts an input and the previous hidden start
        # At the start, the first input is the SOS token and the
        # hidden state is the output of the encoder i.e. context vector

        #The first input of the decode is the SOS token.
        #Iterate through the batch of A_tensors and get the first element in each sequence
        decoder_input = A_tensors[:, 0].view(-1,1) #decoder_input (batch_size, 1)

        #The initial hidden input is the output of the encoder.
        #For each sequence in the encoder output batch, get the last token
        decoder_hidden = output[:,-1:,:].squeeze(1) #decoder_hidden  (batch_size ,sizeof_hidden)

        #Get the output from the decoder.
        #decoder_input (batch_size,1), decoder_hidden (1, batch_size, sizeof_hidden)
        decoder_hidden = decoder_hidden.unsqueeze(0)
        output, hidden = decoder(decoder_input,decoder_hidden) #output (batch_size, 1, sizeof_vocab), hidden (batch_size, sizeof_hidden)
        output = output.squeeze(1)
        #calculate the loss by comparing with the tensor of the first non-sos token
        loss = criterion(output,A_tensors[:,1])
        loss_dialogue = 0
        loss_dialogue += loss

        #Get the top prediction indices
        output = output.topk(k=1, dim=1).indices
        #  For each batch, we now iterate through the rest of the sequence
        # in each A_tensor decoding outputs and hidden states
        for i in range(1,corpus.max_seq_length+1):

            #Get the decoder output and hidden state for the
            output, hidden = decoder(output, hidden)

            output = output.squeeze(1)
            # calculate the loss by comparing with the tensor of the first non-sos token
            target = A_tensors[:, i+1]
            loss = criterion(output, target)
            loss_dialogue += loss

            #Get the top prediction
            output = output.topk(k=1, dim=1).indices
        loss_dialogue = loss_dialogue/12
        loss_batch +=loss_dialogue

        if not current_batch % process_before_update:
            current_batch = 0
            loss_batch = loss_batch / process_before_update
            epoch_loss += loss_batch.item()
            processed_total_batches += 1
            print('Loss={0:.6f}, total phrase pairs in the batch = {1:d}'.format(loss_batch, total_phrase_pairs))
            loss_batch.backward()
            optimizer.step()

    epoch_loss = epoch_loss / (processed_total_batches + 1)
    print('Loss={0:.6f}, total phrase pairs in the batch = {1:d}, total batches processed = {2:d}'.format(epoch_loss,
                                                                                                          total_phrase_pairs,
                                                                                                          processed_total_batches))

    teacher_forcing = max(0,teacher_forcing - (teacher_forcing_decay * teacher_forcing)) #return 0 if negative


# In[ ]:


epoch_losses = np.array(epoch_loss)
plt.plot(epoch_losses[:,0][::1], epoch_losses[:,1][::1])
plt.show()


# In[ ]:


print(epoch_losses[:10,0][::2])


# In[ ]:




