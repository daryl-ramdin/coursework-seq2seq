import os
import re
import nltk

class Vocabulary:
    def __init__(self):
        self.vocabulary = {}
        self.vocabulary_length = 0

    def addWords(self, sentence):
        # Iterates through the tokens in the sentence and updates the vocabulary
        tokens = sentence.lower().split()
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = {"index": self.vocabulary_length, "count": 1}
                self.vocabulary_length += 1
            else:
                self.vocabulary[token]["count"] = self.vocabulary[token]["count"] + 1

    def phraseToVec(self, phrase):
        # Converts a phrase to a list of vectors of its tokens
        # Ensure the phrase is lower case
        phrase = phrase.lower()
        vectorList = []
        vector_template = [0 for i in range(0,self.vocabulary_length)]
        for token in phrase.split():
            #Create a copy of the vector template and set the 
            #token's index to 1
            token_vector = vector_template.copy()
            token_vector[self.vocabulary[token]["index"]] = 1
            vectorList.append(token_vector)
        return vectorList

class Corpus:
    def __init__(self):
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
                

class CornellMovieCorpus(Corpus):
    def __init__(self):
        super().__init__()

        #Let's load the movie lines
        self.movie_lines = self.load_movie_lines()

        #Let's load the conversations
        self.movie_convos = self.load_movie_conversations()

        #get the pairs of conversation exchanges
        self.movie_exchange_pairs = self.create_exchange_pairs()
    
    def create_vocabulary(self):
        #Iterate through the list of movie lines and update the words in the vocabulary
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
        
        # We must now iterate through each conversation and create pairs of exchanges.
        print("Creating exchange pairs")
        movie_exchange_pairs = []
        for convo in movie_convo_lines:
            # Each convo is a list of length > 2
            for i in range(len(convo) - 1):
                movie_exchange_pairs.append((convo[i], convo[i + 1]))

        return movie_exchange_pairs
            
            

    def load_movie_lines_old(self):
        # Movie lines.
        # ref: https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus?resource=download
        # ref: INM706 Lab 5
        # This is tab separated file in the format in the format:
        # lineID
        # characterID (who uttered this phrase)
        # movieID
        # character name
        # text of the utterance
        movie_lines = []
        movie_lines_path = os.path.join("data", "movie_lines.txt")
        with open(movie_lines_path, 'r', encoding="iso-8859-1") as lines:
            for line in lines:
                line_items = line.split(' +++$+++ ')
                movie_lines.append(line_items)
                
        return movie_lines

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

        for i in range(10):
            print(movie_convos[i])

        return movie_convos

    def create_exchange_pairs_old(self):
        # We need to convert our conversations to text.
        # ref: INM706 Lab 5
        print("Creating dictionary of utterances...")
        self.movie_utterances = {}
        for i, line in enumerate(self.movie_lines):
            self.movie_utterances[line[0]] = self.data_prep(line[4].strip())

        print("Converting conversation line numbers to texts...")
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
            convo_lines = [self.movie_utterances[lineid] for lineid in convo]
            movie_convo_lines.append(convo_lines)
        
        # We must now iterate through each conversation and create pairs of exchanges.
        print("Creating exchange pairs")
        movie_exchange_pairs = []
        for convo in movie_convo_lines:
            # Each convo is a list of length > 2
            for i in range(len(convo) - 1):
                movie_exchange_pairs.append((convo[i], convo[i + 1]))

        for i in range(10):
            print(movie_exchange_pairs[i])

        return movie_exchange_pairs
    
    def create_vocabulary_old(self):
        print("Creating vocabulary...")
        
        #Iterate through the words in the movie lines and add them to the 
        #vocabulary only once
        for lineid in self.movie_utterances.keys():
            self.vocabulary_add(self.movie_utterances[lineid])
        
        for i in range(len(self.vocabulary)):
            self.vocabulary[i] = (i,self.vocabulary[i])

