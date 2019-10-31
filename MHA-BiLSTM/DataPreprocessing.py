import numpy as np
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import re
import os
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader, Dataset, TensorDataset
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from torch import tensor
from torch import int32, float32
from pprint import pprint
import pandas as pd
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re


## variable config change ##
dataDistributionType="Balanced"
SheetName="balanced"
embeddingName="glove"
dim="100d"
embedding_dim=100
## path variables ##
rootDir="Your Root Dir"
inputRawData=rootDir+"InputRawData"
InputCleanData=rootDir+"InputCleanData/"
PreTrainVectors=rootDir+"PreTrainVectors/"
NumpyFile=rootDir+"NumpyFile"

RawInputTrainFile=dataDistributionType+"_Train.xlsx"
RawInputTestFile=dataDistributionType+"_Test.xlsx"
CleanTrainFile=dataDistributionType+"_Train.tsv"
CleanTestFile=dataDistributionType+"_Test.tsv"

## numpy file generation 
stored_vectors=NumpyFile+"/"+dataDistributionType+"_"+embeddingName+"_"+dim+".npy"
word_embedding_raw_file=PreTrainVectors+"/"+embeddingName+"/"+embeddingName+"_"+dim+".txt"


train_filepath=os.path.join(inputRawData,RawInputTrainFile)
test_filepath=os.path.join(inputRawData,RawInputTestFile)
df_train = pd.read_excel(train_filepath, sheet_name=SheetName)
df_test = pd.read_excel(test_filepath, sheet_name=SheetName)
#df_test
class Review:
    def __init__( self, review_id, review_text, label= None ):
            
        self.review_id = review_id
        self.review_text = review_text
        self.label = label
        
        self.tokenized_text = [] # tokenized after the Dataset is parsed by the Vocab
        
    
    def __str__(self):
        return str(self.__dict__)
    
    def set_tokenized_text(self, tokenized_text):
        self.tokenized_text = tokenized_text
class Vocab:

    def __init__( self, texts ):
        """
        :type texts: list of strings or Records
        :param texts: text of the reviews is either directly given or is extracted from the objects 
        Assumption: list contains elements of one kind alone
        """

        self.word_to_idx = dict()
        self.id_to_word = dict()
        self.size_of_vocab = 0
        self.words=[]
        self.word_vocab = set()
        self.tokenizer = word_tokenize
        if isinstance(texts, pd.DataFrame):
          cnt=0
          print("Length is ", len(texts))
          for i in texts['comment']:
            
            cnt =cnt+1
            try:
              
              self.words = self.tokenizer(i)
              self.word_vocab.update(set(self.words))
            except Exception as e:
              print('Exception : ',i)  
            if not (cnt%10000):
              print("No. Of sentence Processed is : ",cnt)
          self.word_vocab.add("<unknown>")
          self.word_vocab.add("<number>")
          self.word_vocab.add("<pad>")
          self.size_of_vocab=len(self.word_vocab)
          print("Vocab Creation DONE")
        
        elif isinstance( texts[0], Review ):
          print("Vocab is build")
          for review in texts:
            self.words = self.tokenizer( review.review_text )
            self.word_vocab.update(set(self.words))
          self.word_vocab.add("<unknown>")
          self.word_vocab.add("<number>")
          self.word_vocab.add("<pad>")
          self.size_of_vocab=len(self.word_vocab)
            
        else:
           raise Exception('input should be a list of stings or a list of Review objects')
        
        for _id,word in enumerate(self.word_vocab):
          self.word_to_idx[word]=_id
          self.id_to_word[_id]=word
        
      
    def print_vocab( self ):
        
        for idx, word in self.word_to_idx.items():
            print(idx, word) 
    
    def get_vocab(self):
        return self.word_to_idx

    def get_vocab_size(self):

        return self.size_of_vocab

    def convert_text_to_sequence_numbers(self, reviews):
        """ 
        :type reviews: list of strings or a string
        :param reviews: review's text
    
        :rtype:  list(list(int)) 
        """    
              
        
        if isinstance(reviews, str):
            review_sequences = []
            for token in self.tokenizer( reviews ):
                # sequence numbers are generated only for those sentences that are present in the vocab
                if token in self.word_to_idx:
                    review_sequences.append( self.word_to_idx[ token] )
                else:
                    review_sequences.append( self.word_to_idx[ '<unknown>' ] )
            return review_sequences
            
     

    def pad_sequence(self, input_sequence, length, padding= 'post', pad_character='<pad>'):
        
        original_length = len( input_sequence )
        if len( input_sequence ) < length:
            if padding == 'post':
                input_sequence = input_sequence + [ self.word_to_idx['<pad>'] ] * ( length - len(input_sequence) )
            elif padding == 'pre':
                input_sequence = [ self.word_to_idx['<pad>'] ] * ( length - len( input_sequence ) ) + input_sequence 
        
        return tensor( input_sequence ), tensor( original_length, dtype= int32)
		

		
def dataPreprocessing(RawFilePath,Save_PreProcessed_File_Path):
  sarcasm_list=[]
  p=0
  df = pd.read_excel(RawFilePath, sheet_name=SheetName)
  for i in df.values:
    review_text=i[0]
    review_id=i[1]
    label=i[2]
    sentence = review_text.strip()
    sentence = re.sub(r'[\t]',' ',sentence)
    for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
      sentence = sentence.replace(ch, " " + ch + " ")
      
    sentence = sentence.replace('\\n'," ")
    sentence = re.sub(r'[^a-zA-Z !@#*\'\":;.,?]', '',sentence).lower()
    
    sentence = sentence.replace('n\'t','')
    sentence = sentence.replace('\'ll','')
    sentence = sentence.replace('\'m','')
    sentence = sentence.replace('\'s','')
    sentence = sentence.replace('\'re','')
    sentence = sentence.replace('\'ve','')
    sentence = sentence.replace('\'m','')
    sentence = sentence.replace('\'d','')
    sentence = sentence.replace('\'ll','')
    
    sentence = ' '.join([w.strip() for w in sentence.split()])
      
    #sentence = ' '.join( sentence.split() )
    sarcasm_list.append(Review(review_id,sentence,label)) 
  with open( Save_PreProcessed_File_Path, 'w' ) as f:
    f.write('id'+ '\t' +'comment'+ '\t' +'label\n')
    for i in sarcasm_list:
      if len(str(i.review_text)) == 0 or len(str(i.review_text)) > 150 or len(str(i.review_text)) < 30:
        #print ("Pruned ",len(str(i.review_text)) )
        p +=1
      else:
        f.write( str(i.review_id) + '\t' + str(i.review_text) + '\t'+str(i.label) )
        f.write('\n')
  print("total discarded sentences are : ",p)
def Train_And_Test_Comments(train_file,test_file ):
  df_train=pd.read_csv(train_file, sep='\t')
  df_test=pd.read_csv(test_file, sep='\t')
  df=pd.concat([df_train,df_test],ignore_index=True)
  return df



def create_embedding_matrix( vocab, embedding_dim,word_embedding_raw_file,save_weight_path):
  num_tokens = vocab.get_vocab_size()
  word2idx = vocab.get_vocab()
  embedding_matrix = np.zeros( ( num_tokens, embedding_dim ) )
  print('number of tokens: ', num_tokens, ' embedding dimensions: ', embedding_dim )
  print('embedding matrix shape: ',embedding_matrix.shape)
  with open(word_embedding_raw_file, 'r',encoding='utf-8') as f:
    num_mapped_words = 0
    num_total_words_seen = 0
    for line in f:
      values = line.split()
      word = values[0]
      num_total_words_seen += 1
      if word in word2idx and not word == '<pad>' and not word == '<unknown>':
        vector = np.asarray( values[1:] )
        embedding_matrix[ word2idx[ word ], : ] = vector
        num_mapped_words += 1
        if num_total_words_seen % 1000000 == 0:
          print('num_total_words_seen ',num_total_words_seen)
       
    print('loaded pretrained matrix, ', ' num mapped words: ', num_mapped_words)
    np.save(save_weight_path, embedding_matrix)
    print('matrix is saved')
	 
def main():

  Clean_Train_Data=os.path.join(InputCleanData+CleanTrainFile)
  Clean_Test_Data=os.path.join(InputCleanData+CleanTestFile)
  dataPreprocessing(train_filepath,Clean_Train_Data)
  dataPreprocessing(test_filepath,Clean_Test_Data)
  CombinedDF =Train_And_Test_Comments(Clean_Train_Data, Clean_Test_Data )
  vocab=Vocab(CombinedDF)
  create_embedding_matrix(vocab,embedding_dim,word_embedding_raw_file,stored_vectors)
if __name__ == "__main__":
    main()		