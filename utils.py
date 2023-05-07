import numpy as np
import torch
from corpus import Corpus, CornellMovieCorpus

def evaluate(sentence, encoder, decoder, corpus, device):
    # To evaluate

    encoder.eval()
    decoder.eval()
    Q_tensor, seq_length = corpus.sentenceToTensor(sentence)

    Q_tensor = Q_tensor.view(-1,1)

    with torch.no_grad():

        encoder_output, encoder_hidden = encoder(Q_tensor, [seq_length])  # encoder_output: (batch_size, max_seq_len, hidden_size), encoder_hidden: (1, batch_size, hidden_size)

        # At the start, the first input to the decoder is the SOS token
        # Create a batch of SOS tensors
        decoder_input = torch.tensor(corpus.vocabulary.SOS_index, dtype=torch.int64, device=device)
        decoder_input = decoder_input.view(-1,1)

        # The initial hidden input to the decoder is the last hidden of the encoder.
        # For each sequence in the encoder hidden batch, get the last token. encoder_hidden: (batch_size, max_seq_len, hidden_size)
        decoder_hidden = encoder_hidden

        reply = ""
        for i in range(seq_length):

            # Get the decoder output and hidden state for the
            #decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_output)  # decoder_output (batch_size, 1, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # decoder_output (batch_size, 1, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)

            decoder_output = decoder_output.squeeze(0)

            # Get the top prediction for each batch
            decoder_input = decoder_output.topk(k=1, dim=1).indices
            reply += corpus.vocabulary.indexWord(decoder_input.item()) + " "

    encoder.train()
    decoder.train()
    print(reply)

def maskNLLLoss(inp, target, mask, device):
    mask = mask.to(device)
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()