import os
import re
import nltk
import random
import torch

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
        #Add the start of sentence and end of sentence
        self.vocabulary[Vocabulary.PAD] = {"index": Vocabulary.PAD_index , "count": 1}
        self.vocabulary[Vocabulary.SOS] = {"index": Vocabulary.SOS_index , "count": 1}
        self.vocabulary[Vocabulary.EOS] = {"index": Vocabulary.EOS_index , "count": 1}
        self.len = 3


    def addWords(self, sentence):
        # Iterates through the tokens in the sentence and updates the vocabulary
        tokens = sentence.lower().split()
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = {"index":self.len, "count":1}
                self.len += 1
            else:
                self.vocabulary[token]["count"] = self.vocabulary[token]["count"] + 1
    
    def indexWord(self,index):
        return [word[0] for word in self.vocabulary.items() if word[1]["index"] == index][0]
    
    
    def wordIndex(self,word):
        return self.vocabulary[word]["index"]

    def wordsToIndex(self, sentence):
        # Converts a sentence to a list of indices of its words
        indicesList = []
        for word in sentence.split():
            indicesList.append(self.vocabulary[word]["index"])
        return indicesList

class Corpus:
    def __init__(self,max_seq_length=10,):
        self.max_seq_length = max_seq_length
        self.exchange_pairs = []
        self.vocabulary = Vocabulary()

    def get_exchange_pairs(self):
        return self.exchange_pairs

    def get_vocabulary(self):
        return self.vocabulary

    def data_prep(self,sentence):
        #remove non-alphanumeric
        #ref INM706 Lab 5
        prepped = re.sub(r"\W+"," ",sentence).strip().lower()
        return prepped
    
    def create_vocabulary(self):
        return
    
    def get_random_exchange(self):
        return random.choice(self.exchange_pairs)
    
    def pairToTensor(self,pair):
        #First convert the pair to list of indices
        #ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        indices_pair = []
        indices_pair.append(self.vocabulary.wordsToIndex(pair[0])) #Input
        indices_pair.append(self.vocabulary.wordsToIndex(pair[1])) #Target
        
        #We then standardize the length of each sequence, truncating
        #those above the length and padding those below
        #ref INM706 Lab5
        i = 0
        for i in range(len(indices_pair)):
            seq_len = len(indices_pair[i])
            
            #Truncate sequence if too long
            if seq_len > self.max_seq_length: indices_pair[i] = indices_pair[i][:self.max_seq_length]

            #Add the EOS
            indices_pair[i].append(Vocabulary.EOS_index)

            #If the length of the original seq is less than the max, then pad
            if seq_len < (self.max_seq_length):
                #ref: https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
                indices_pair[i] += [Vocabulary.PAD_index] * (self.max_seq_length - seq_len)
            
            #Insert the SOS index
            indices_pair[i].insert(0,Vocabulary.SOS_index)

                                            
        input_tensor = torch.tensor(indices_pair[0], dtype=torch.int64).view(-1,1)
        target_tensor = torch.tensor(indices_pair[1], dtype=torch.int64).view(-1,1)
        return input_tensor, target_tensor

    def wordToTensor(self,word):
        word_tensor = torch.tensor(self.vocabulary.wordsToIndex(word), dtype=torch.int64).view(-1,1)
        return word_tensor
                

class CornellMovieCorpus(Corpus):
    def __init__(self):
        super().__init__()

        #Let's load the movie lines
        self.movie_lines = self.load_movie_lines()

        #Let's load the conversations
        self.movie_convos = self.load_movie_conversations()

        #get the pairs of conversation exchanges
        self.exchange_pairs = self.create_exchange_pairs()
    
    def create_vocabulary(self):
        #Iterate through the list of movie lines and update the words in the vocabulary
        self.vocabulary = Vocabulary()
        for sentence in self.movie_lines.values():
            self.vocabulary.addWords(sentence["prepped_text"])
        
    def load_movie_lines(self):
        #Let's load the movie lines. For each line we 
        #will store the original text as well as the prepped text
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
                movie_lines[line_items[0]] = {"original_text":sentence,"prepped_text":self.data_prep(sentence)}
                
        return movie_lines
    
    def create_exchange_pairs(self):
        # We need to convert our conversations to text.
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
            
            # ref: https://stackoverflow.com/questions/4071396/how-to-split-by-comma-and-strip-white-spaces-in-python
            convo_lines = [self.movie_lines[lineid]["prepped_text"] for lineid in convo]
            movie_convo_lines.append(convo_lines)
            
            if i==5000: break
        
        # We must now iterate through each conversation and create pairs of exchanges.
        print("Creating exchange pairs")
        movie_exchange_pairs = []
        for convo in movie_convo_lines:
            # Each convo is a list of length > 2
            for i in range(len(convo) - 1):
                movie_exchange_pairs.append((convo[i], convo[i + 1]))

        return movie_exchange_pairs
            
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
