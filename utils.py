import numpy as np
import torch
from corpus import Corpus, CornellMovieCorpus

def evaluate(sentence, encoder, decoder, corpus):
    # To evaluate
    seq_length = corpus.max_seq_length

    Q_tensor = corpus.sentenceToTensor(sentence)

    Q_tensor = Q_tensor.view(1,-1)

    Q_tensor = Q_tensor.to(device)

    with torch.no_grad():
        encoder_output, encoder_hidden = encoder(Q_tensor)  # encoder_output: (batch_size, max_seq_len, hidden_size), encoder_hidden: (1, batch_size, hidden_size)

        # The first input to the decoder is the SOS token.
        # Iterate through the batch of A_tensors and get the first element in each sequence
        decoder_input = torch.tensor(corpus.vocabulary.SOS_index, dtype=torch.int64)  # decoder_input (batch_size, 1)

        decoder_input = decoder_input.view(1,-1)
        # The initial hidden input to the decoder is the last hidden of the encoder.
        # For each sequence in the encoder hidden batch, get the last token. encoder_hidden: (batch_size, max_seq_len, hidden_size)
        decoder_hidden = encoder_hidden

        reply = ""
        for i in range(1,seq_length):

            # Get the decoder output and hidden state for the
            #decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_output)  # decoder_output (batch_size, 1, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # decoder_output (batch_size, 1, sizeof_vocab), decoder_hidden (batch_size, sizeof_hidden)

            decoder_output = decoder_output.squeeze(1)

            # Get the top prediction for each batch
            decoder_input = decoder_output.topk(k=1, dim=1).indices
            reply += corpus.vocabulary.indexWord(decoder_input.item()) + " "

    print(reply)
