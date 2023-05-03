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


random.seed(45)

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
pairs, batch = corpus.getBatchExchangeTensors(5)

for i in range(len(pairs)):
    print(pairs[i]["Q"],"\n")
    print(batch[0][i],"\n")
    print(pairs[i]["A"],"\n")
    print(batch[1][i],"\n")


# <h2>Create Encoders and Decoders</h2>

# In[6]:


sizeof_embedding = 256
sizeof_vocab = corpus.vocabulary.len

encoder = Encoder(sizeof_vocab, sizeof_embedding)
decoder = Decoder(sizeof_embedding, sizeof_vocab)


# <h2>Let's setup our trainer</h2>

# In[7]:


number_of_epochs = 75000
print_interval = 10
batch_size = 10
teacher_forcing = 0
teacher_forcing_decay = 0

encoder_optimizer = optim.Adam(encoder.parameters(),lr=1e-03)
decoder_optimizer = optim.Adam(decoder.parameters(),lr=1e-03)

criterion = nn.NLLLoss()

#Let's get a random exchange pair
epoch_loss = []
start_time = datetime.now()
for epoch in range(number_of_epochs):
    
    pairs, batch = corpus.getBatchExchangeTensors(5)
    
    Q_tensors = batch[0]
    A_tensors= batch[1]
    print("Tensor shapes", Q_tensors.shape, A_tensors.shape)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #input_tensor, target_tensor = corpus.pairToTensor(exchange_pair)
    #print("Input tensor shape", input_tensor.shape, "target", target_tensor.shape)

    #Try this on the encoder
    #We need to initialise the hidden state
    hidden = torch.zeros(1,corpus.max_seq_length,encoder.sizeof_embedding)

    encoder_output = []

    #Encode each word in the input tensor one word a time
    for input_tensor in Q_tensors:
        output, hidden = encoder(input_tensor,hidden)
        #We also keep an array of the outputs
        encoder_output.append(output)

    # #The decoder accepts an input and the previous hidden start
    # #At the start, the first input is the SOS token and the 
    # #previous hidden state is the output of the encoder i.e. context vector

    int_t = torch.tensor(Vocabulary.SOS_index,dtype=torch.int64)
    
    hidden = encoder_output[len(encoder_output)-1]

    loss = 0
    

    # print("Target Tensor", target_tensor.shape)

    decoder_output = []
    
    for i in range(len(target_tensor)):
        if random.random() < teacher_forcing: int_t = target_tensor[i]
        
        output, hidden = decoder(int_t,hidden)
        #print("Output", output.shape, "Target", target_tensor[i].shape)
        loss += criterion(output,target_tensor[i])
        int_t = torch.argmax(output,dim=1)
            
    if epoch%print_interval == 0: 
        end_time = datetime.now()
        timediff = end_time - start_time
        print("Epoch", epoch, "Loss", loss.item()/len(target_tensor), "teacher_forcing", teacher_forcing, "in",timediff.seconds, "seconds") 
        start_time = datetime.now()
        
    epoch_loss.append([epoch,loss.item()/len(target_tensor)])
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    teacher_forcing = max(0,teacher_forcing - (teacher_forcing_decay * teacher_forcing)) #return 0 if negative


# In[ ]:


epoch_losses = np.array(epoch_loss)
plt.plot(epoch_losses[:,0][::500], epoch_losses[:,1][::500])
plt.show()


# In[ ]:


print(epoch_losses[:10,0][::2])


# In[ ]:




