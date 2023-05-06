import os
import re
import nltk
import random
import torch
from torch.utils.data import Dataset


class Vocabulary:
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    PAD_index = 0
    SOS_index = 1
    EOS_index = 2

    def __init__(self):
        self.vocabulary = {}
        self.len = 0
        # Add the start of sentence and end of sentence
        self.vocabulary[Vocabulary.PAD] = {"index": Vocabulary.PAD_index, "count": 1}
        self.vocabulary[Vocabulary.SOS] = {"index": Vocabulary.SOS_index, "count": 1}
        self.vocabulary[Vocabulary.EOS] = {"index": Vocabulary.EOS_index, "count": 1}
        self.len = 3

    def addWords(self, sentence):
        # Iterates through the tokens in the sentence and updates the vocabulary
        tokens = sentence.lower().split()
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = {"index": self.len, "count": 1}
                self.len += 1
            else:
                self.vocabulary[token]["count"] = self.vocabulary[token]["count"] + 1

    def indexWord(self, index):
        return [word[0] for word in self.vocabulary.items() if word[1]["index"] == index][0]

    def wordIndex(self, word):
        return self.vocabulary[word]["index"]

    def wordsToIndex(self, sentence):
        # Converts a sentence to a list of indices of its words
        indicesList = []
        for word in sentence.split():
            indicesList.append(self.vocabulary[word]["index"])
        return indicesList


class Corpus(Dataset):
    FULL = 1
    PAIRS = 2

    def collate_convos(batch):
        return batch
    def __init__(self, max_seq_length=10, convo_mode=FULL):
        super(Corpus, self).__init__()
        self.max_seq_length = max_seq_length
        self.conversations = []
        self.vocabulary = Vocabulary()
        self.convo_mode = convo_mode

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, item):
        return item, self.get_conversation(item)

    def get_conversation(self, index):
        batch = []
        Q_tensors = []
        A_tensors = []

        # Get the batch of pairs. It encapsulates the conversation in a list
        conversation = self.conversations[index]

        #If our mode is FULL then we get a list of exchange pairs
        #otherwise it is an single exchange pair. If mode is PAIRS
        #encapsulate as a list of standardised iteration after
        if self.convo_mode==Corpus.PAIRS:
            conversation = [conversation]


        # Convert each pair to a tensor of its token indices
        for exchange_pair in conversation:
            q, a = self.pairToTensor(exchange_pair)
            Q_tensors.append(q)
            A_tensors.append(a)

        # ref INM706 Lab 5
        Q_tensors = torch.stack(Q_tensors)
        A_tensors = torch.stack(A_tensors)
        return {"conversation":conversation, "Q":Q_tensors, "A":A_tensors}

    def data_prep(self, sentence):
        # remove non-alphanumeric
        # ref INM706 Lab 5
        prepped = re.sub(r"\W+", " ", sentence).strip().lower()
        return prepped

    def create_vocabulary(self):
        return

    def get_random_conversation(self):
        return random.choice(self.conversations)

    def pairToTensor(self, pair):
        # First convert the pair to list of indices
        # ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        QandA = []
        QandA.append(pair["Q"]["indices"].copy())  # Input
        QandA.append(pair["A"]["indices"].copy())  # Target

        # We then standardize the length of each sequence, truncating
        # those above the length and padding those below
        # ref INM706 Lab5
        i = 0
        for i in range(2):
            seq_len = len(QandA[i])

            # Truncate sequence if too long
            if seq_len > self.max_seq_length: QandA[i] = QandA[i][:self.max_seq_length]

            # Add the EOS
            QandA[i].append(Vocabulary.EOS_index)

            # If the length of the original seq is less than the max, then pad
            if seq_len < (self.max_seq_length):
                # ref: https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
                QandA[i] += [Vocabulary.PAD_index] * (self.max_seq_length - seq_len)

            # Insert the SOS index
            QandA[i].insert(0, Vocabulary.SOS_index)

        Q_tensor = torch.tensor(QandA[0], dtype=torch.int64)
        A_tensor = torch.tensor(QandA[1], dtype=torch.int64)
        return Q_tensor, A_tensor

    def wordToTensor(self, word):
        word_tensor = torch.tensor(self.vocabulary.wordsToIndex(word), dtype=torch.int64).view(-1, 1)
        return word_tensor

    def sentenceToTensor(self,sentence):
        #Prep the sentence
        sentence = self.data_prep(sentence)

        #Get the list of indices
        indices = self.vocabulary.wordsToIndex(sentence)

        # We then standardize the length of each sequence, truncating
        # those above the length and padding those below
        # ref INM706 Lab5

        seq_len = len(indices)

        # Truncate sequence if too long
        if seq_len > self.max_seq_length: indices = indices[:self.max_seq_length]

        # Add the EOS
        indices.append(Vocabulary.EOS_index)

        # If the length of the original seq is less than the max, then pad
        if seq_len < (self.max_seq_length):
            # ref: https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
            indices += [Vocabulary.PAD_index] * (self.max_seq_length - seq_len)

        # Insert the SOS index
        indices.insert(0, Vocabulary.SOS_index)

        Q_tensor = torch.tensor(indices, dtype=torch.int64)

        return Q_tensor

class CornellMovieCorpus(Corpus):
    def __init__(self, max_seq_length=10, convo_mode=Corpus.FULL):
        super(CornellMovieCorpus, self).__init__(max_seq_length, convo_mode)

        # Let's load the movie lines
        self.movie_lines = self.load_movie_lines()

        # Let's load the conversations
        self.movie_convos = self.load_movie_conversations()

        # create the vocabulary
        self.create_vocabulary()

        # Depending on the mode we either create pairs of exchanges
        # or full conversations
        if self.convo_mode == Corpus.FULL:
            # We are storing full conversations
            self.conversations = self.create_conversation_chains()
        else:
            # We are storing exchange pairs
            self.conversations = self.create_exchange_pairs()

    def create_vocabulary(self):
        # Iterate through the list of movie lines and update the words in the vocabulary
        print("Creating vocabulary...")
        self.vocabulary = Vocabulary()
        for sentence in self.movie_lines.values():
            self.vocabulary.addWords(sentence["prepped_text"])

    def load_movie_lines(self):
        # Let's load the movie lines. For each line we
        # will store the original text as well as the prepped text
        # ref: https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus?resource=download
        # ref: INM706 Lab 5
        # This is tab separated file in the format in the format:
        # lineID
        # characterID (who uttered this phrase)
        # movieID
        # character name
        # text of the utterance
        print("Loading movie lines...")
        movie_lines = {}
        filepath = os.path.join("data", "movie_lines.txt")
        with open(filepath, 'r', encoding="iso-8859-1") as lines:
            for line in lines:
                line_items = line.split(' +++$+++ ')
                sentence = line_items[-1]
                movie_lines[line_items[0]] = {"original_text": sentence, "prepped_text": self.data_prep(sentence)}

        return movie_lines

    def convolines2text(self):
        # ref: INM706 Lab 5
        print("Converting conversation line numbers to text...")
        movie_convo_lines = []
        for i, convo in enumerate(self.movie_convos):
            # Get the conversation
            convo = convo[-1]

            # Remove the square brackets and get each line id
            convo = convo[1:len(convo) - 1].split(",")

            # Line ids have spaces so remove
            convo = [lineid.strip() for lineid in convo]

            # The'yre also encapsualted in quotes so remove them
            convo = [lineid[1:len(lineid) - 1] for lineid in convo]

            # Our convo is of the format [L1, L2, L3...]

            # ref: https://stackoverflow.com/questions/4071396/how-to-split-by-comma-and-strip-white-spaces-in-python
            convo_lines = [self.movie_lines[lineid]["prepped_text"] for lineid in convo]
            movie_convo_lines.append(convo_lines)
        return movie_convo_lines

    def create_exchange_pairs(self):
        # We need to convert our conversations to text.
        movie_convo_lines = self.convolines2text()

        # We must now iterate through each conversation and create pairs of exchanges.
        print("Creating exchange pairs")
        movie_exchange_pairs = []
        movie_convo_pairs = []
        for convo in movie_convo_lines:
            # Each convo is a list of length > 2
            for i in range(len(convo) - 1):
                q = {"tokens": convo[i], "indices": self.vocabulary.wordsToIndex(convo[i])}
                a = {"tokens": convo[i + 1], "indices": self.vocabulary.wordsToIndex(convo[i + 1])}
                movie_exchange_pairs.append({"Q": q, "A": a})

        return movie_exchange_pairs

    def create_conversation_chains(self):
        # We need to convert our conversations to text
        # and then create a chain of links. Each
        # link is an exchange pair between two people
        # ref: INM706 Lab 5
        # We need to convert our conversations to text.
        movie_convo_lines = self.convolines2text()

        # We must now iterate through each conversation and create pairs of exchanges.
        print("Creating conversation chain")
        conversation_chains = []
        for convo in movie_convo_lines:
            # Each convo is a list of length > 2
            convo_chain = []
            for i in range(len(convo) - 1):
                q = {"tokens": convo[i], "indices": self.vocabulary.wordsToIndex(convo[i])}
                a = {"tokens": convo[i + 1], "indices": self.vocabulary.wordsToIndex(convo[i + 1])}
                convo_chain.append({"Q": q, "A": a})
            conversation_chains.append(convo_chain)

        return conversation_chains

    def load_movie_conversations(self):
        # Let's load the conversations.
        # ref: https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus?resource=download
        # ref: INM706 Lab 5
        # They are in the format:
        # characterID of the first character involved in the conversation
        # characterID of the second character involved in the conversation
        # movieID of the movie in which the conversation occurred
        # list of the utterances that make the conversation, in chronological
        # order: ['lineID1','lineID2',É,'lineIDN']
        movie_convos = []
        movie_convos_path = os.path.join("data", "movie_conversations.txt")
        with open(movie_convos_path, 'r', encoding="iso-8859-1") as convos:
            for convo in convos:
                movie_convos.append(convo.strip().split(' +++$+++ '))

        return movie_convos

