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
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dill

CUDA=torch.cuda.is_available()
device=torch.device("cuda" if CUDA else "cpu")
use_cuda = torch.cuda.is_available()
DEVICE=torch.device("cuda:0" if use_cuda else "cpu")

## variable config change ##
dataDistributionType="Balanced"
SheetName="balanced"
embeddingName="glove"
dim="100d"
## path variables ##
rootDir="Your_Root_Dir"
inputRawData=rootDir+"InputRawData"
InputCleanData=rootDir+"InputCleanData/"
PreTrainVectors=rootDir+"PreTrainVectors/"
NumpyFile=rootDir+"NumpyFile"
Results="result"
BestModel="BestModel"
output="output/"
RawInputTrainFile=dataDistributionType+"_Train.xlsx"
RawInputTestFile=dataDistributionType+"_Test.xlsx"
CleanTrainFile=dataDistributionType+"_Train.tsv"
CleanTestFile=dataDistributionType+"_Test.tsv"
output_file_path=rootDir+output+dataDistributionType+"_output.tsv"

## numpy file generation 
stored_vectors=NumpyFile+"/"+dataDistributionType+"_"+embeddingName+"_"+dim+".npy"
word_embedding_raw_file=PreTrainVectors+"/"+embeddingName+"/"+embeddingName+"_"+dim+".txt"

## result file path
ResultfilePath=rootDir+Results
BestModelDir=rootDir+BestModel

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


def Train_And_Test_Comments(train_file,test_file ):
  df_train=pd.read_csv(train_file, sep='\t')
  df_test=pd.read_csv(test_file, sep='\t')
  df=pd.concat([df_train,df_test],ignore_index=True)
  return df	

Clean_Train_Data=os.path.join(InputCleanData+CleanTrainFile)
Clean_Test_Data=os.path.join(InputCleanData+CleanTestFile)
CombinedDF =Train_And_Test_Comments(Clean_Train_Data, Clean_Test_Data )
vocab=Vocab(CombinedDF)

class ReviewDataset(Dataset):
    def __init__( self, dataset_path, device=DEVICE , preprocessed= False, vocab= None ):
    
        self.device = device
        self.max_review_length = -1
        self.review_list = []
        self.labels =[]
        df = pd.read_csv(dataset_path, sep='\t')
        for line in df.values:
          review_text = line[1]
          review_id = line[0]
          label = line[2] if len( line ) > 0 else None
                    
          self.review_list.append( Review( review_id, review_text, label ) )
        print('loading file complete')
        
        self.tokenizer = Vocab( self.review_list ) if vocab == None else vocab
        
        for review in self.review_list:
          
          review.tokenized_text = self.tokenizer.convert_text_to_sequence_numbers( review.review_text )
          self.labels.append (review.label)
          #print(review.tokenized_text)
          self.max_review_length = max( len( review.tokenized_text ), self.max_review_length )

       
    def get_vocab(self):
        return self.tokenizer

    def get_review_list(self):
        return self.review_list

    def __len__(self):
        return len( self.review_list )

    def __getitem__(self, idx):

        data_item = self.review_list[idx]
        padded_review, original_review_length = self.tokenizer.pad_sequence( data_item.tokenized_text, self.max_review_length )
        #padded_review, original_review_length = data_item.tokenized_text, len( data_item.tokenized_text )
        labels = data_item.label
        review_id=data_item.review_id
        item = {    
                    'review': padded_review,
                    'original_review_length': original_review_length,
                    'targets': labels,
                    'id':review_id
                }

        return item  
		
class BiLSTM(nn.Module):
  def __init__(self,weight,embedding_dim,hidden_dim,output_dim):
    super(BiLSTM,self).__init__()
    self.embedding=nn.Embedding.from_pretrained(weight)
    self.hidden_dim = hidden_dim
    self.output_dim=output_dim
    self.dropout=nn.Dropout(0.3)
    self.bidirectional=True
    self.lstm=nn.LSTM(embedding_dim,
                     hidden_dim,
                     num_layers=1,
                     batch_first=True,
                     bidirectional =self.bidirectional,
                     dropout=0.0)
    self.fc=nn.Linear(2*hidden_dim,output_dim)
    self.weight_init()
  
    #self.w_r=nn.Linear(1*2*self.hidden_dim,self.output_dim)
    #intializing the weights with uniformity 
  
  def weight_init(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
  def forward(self,input):
    review_lengths=input['original_review_length']
    embedded =self.embedding(input['review'])
    embedded = pack_padded_sequence(embedded, review_lengths, batch_first= True, enforce_sorted= False)
    output,(hidden,cell)=self.lstm(embedded)
    # combine both directions
    if self.bidirectional:
      combined = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
    else:
      combined = hidden
    #combined = self.dropout(combined)
    out=self.fc(combined.squeeze(0))
    return  out.view(-1),cell
	
class MultiHopAttBiLSTM(nn.Module):
  def __init__(self,weight,embedding_dim,hidden_dim,output_dim,batch_size):
    super(MultiHopAttBiLSTM,self).__init__()
    self.embedding=nn.Embedding.from_pretrained(weight)
    self.hidden_dim = hidden_dim
    self.output_dim=output_dim
    self.batch_size=batch_size
    self.d_a=200
    self.r = 20
    self.h_l=100
    self.linear_first = nn.Linear(2*hidden_size, self.d_a)
    self.linear_second = nn.Linear(self.d_a,self.r)
    self.linear_final = nn.Linear(self.r*2*hidden_size, self.h_l)
    self.label = nn.Linear(self.h_l, self.output_dim)
    self.dropout=nn.Dropout(0.5)
    self.bilstm=nn.LSTM(embedding_dim,
                     hidden_dim,
                     num_layers=1,
                     batch_first=True,
                     bidirectional =True,
                     dropout=0.0)
    self.h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
    self.c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
    
  def weight_init(self):
    for p in self.parameters():
      if p.requires_grad:
        if len(p.shape) > 1:
          torch.nn.init.xavier_uniform_(p)
        else:
          stdv = 1. / (p.shape[0] ** 0.5)
          torch.nn.init.uniform_(p, a=-stdv, b=stdv)
  def attention_net(self, lstm_output):
      #lstm_output --> batch_size*Seq_Size*Hidden_size
    lf=self.linear_first(lstm_output)
    act=F.tanh(lf)
    attn_weight_matrix = self.linear_second(act)                                       # batch_size*Seq_Size*r
    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)                           # batch_size*r*Seq_Size
    attn_weight_matrix = torch.nn.functional.softmax(attn_weight_matrix, dim=2)        # batch_size*r*Seq_Size
    return attn_weight_matrix     
    
  
    
  
    
  def forward(self,input):
       
    h=self.h_0
    c=self.c_0
    review_lengths=input['original_review_length']
    embedded =self.embedding(input['review'])
    embedded=self.dropout(embedded)
    
    embedded = pack_padded_sequence(embedded, review_lengths, batch_first= True, enforce_sorted= False)
    (output, (h, c)) = self.bilstm(embedded, (h, c))                                           #(batch_size, Seq_Size, 2*hidden_size)
    output, _ = pad_packed_sequence( output, batch_first= True, padding_value= 0.0 )           #(batch_size,seq_len,hidden_dim)
    
    attn_weight_matrix = self.attention_net(output)                                            #(batch_size, r,   Seq_Size)
    hidden_matrix = torch.bmm(attn_weight_matrix, output)                                      #(batch_size, r,   2*hidden_size)
	
    # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
	
    hidden_matrix=hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])      #(batch_size,r*2*hidden_size)
    sentence_embedding = self.linear_final(hidden_matrix)                                      #(batch_size,self.h_l)
    out    = self.label(sentence_embedding)
    return  out.view(-1),attn_weight_matrix
	
class MultiHeadAttBiLSTM(nn.Module):
  def __init__(self,weight,embedding_dim,hidden_dim,output_dim,batch_size):
    super(MultiHeadAttBiLSTM,self).__init__()
    self.embedding=nn.Embedding.from_pretrained(weight)
    self.hidden_dim = hidden_dim
    self.output_dim=output_dim
    self.d_a=200
    self.r = 4
    self.linear_first = torch.nn.Linear(2*hidden_dim,self.d_a)
    self.linear_first.bias.data.fill_(0)
    self.linear_second = torch.nn.Linear(self.d_a,self.r )
    self.linear_second.bias.data.fill_(0)
    self.n_classes = 1
    self.linear_final = torch.nn.Linear(2*hidden_dim,self.n_classes)
    self.batch_size = batch_size       
    #self.max_len = max_len
    self.num_layers=1
    self.hidden_state = self.init_hidden()
   
    #self.type = type

    self.dropout=nn.Dropout(0.5)
    self.lstm=nn.LSTM(embedding_dim,
                     hidden_dim,
                     num_layers= self.num_layers,
                     batch_first=True,
                     bidirectional =True,
                     dropout=0.0)
    self.fc=nn.Linear(1*2*hidden_dim,output_dim)
    #self.num_heads=5
    # self Attention weight and bais definition 
    #self.weight_m=nn.Parameter(torch.rand(self.num_heads,1*2*self.hidden_dim,1*2*self.hidden_dim))
    self.att_bais=nn.Parameter(torch.rand(1))
    #self.w_t=nn.Linear(self.num_heads,1)
    self.w_r=nn.Linear(self.hidden_dim*2,output_dim)
    self.weight_init()
  def init_hidden(self):

        return (Variable(torch.zeros(2* self.num_layers,self.batch_size,self.hidden_dim)).cuda(),Variable(torch.zeros(2* self.num_layers,self.batch_size,self.hidden_dim)).cuda())
    #intializing the weights with uniformity 
  def weight_init(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
  def forward(self,input):
       
    h_state=self.hidden_state
    review_lengths=input['original_review_length']
    embedded =self.embedding(input['review'])
    embedded=self.dropout(embedded)
    embedded = pack_padded_sequence(embedded, review_lengths, batch_first= True, enforce_sorted= False)
    output,(h_state)=self.lstm(embedded,h_state)
    output, _ = pad_packed_sequence( output, batch_first= True, padding_value= 0.0 ) #(batch_size,seq_len,hidden_dim)
    x = F.relu(self.linear_first(output))    #(batch_size,seq_len,d_a)
    x = self.linear_second(x)                #(batch_size,seq_len,r)
    x = torch.nn.functional.softmax( x,dim =1)   #(batch_size,seq_len,r)        
    attention = x.transpose(1,2)               #(batch_size,r,seq_len)
    sentence_embeddings = attention@output    #(batch_size,r,hidden_dim)
    avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r #(batch_size,1,hidden_dim)
    avg_sentence_embeddings = self.dropout(avg_sentence_embeddings)
    out=self.fc(avg_sentence_embeddings)  #(batch_size,1,output_dim)
    return  out.view(-1),attention 	
def load_pre_trained_embedded_matrix(stored_vector_path,device):
  embedding_weights=np.load(stored_vector_path)
  return torch.from_numpy(embedding_weights).float().to(device)

weights=load_pre_trained_embedded_matrix(stored_vectors,DEVICE)

def SaveBestResult(matrix,filePath,modelName):
  Save_PreProcessed_File_Path=filePath+"/"+modelName+"_"+dataDistributionType+".txt"
  with open( Save_PreProcessed_File_Path, 'w' ) as f:
    f.write('modelName'+ '\t' + 'precision'+ '\t' +'recall'+ '\t' +'f1_score\n')
    f.write( str(modelName) + '\t' + str(matrix["precision"]) + '\t'+str(matrix["recall"]) + '\t'+str(matrix["f1_score"]))
	
def compute_binary_accuracy(model, data_loader, device):
    model.eval()
    tp=0
    tn=0
    fp=0
    fn=0
    val_losses=[]
    best_loss=100
    correct_pred, num_examples = 0, 0
    
    #path_save_best_model=os.path.join(rootDir,"model_weights_bilstm.pt")
    best_f1=0
    
    val_curr_loss=0
    with torch.no_grad():
        #for batch_idx, batch_data in enumerate(data_loader):
        for batch_idx, batch in enumerate(data_loader):
          batch = { k : v.to( device ) for k,v in batch.items() }  
          logits,_ = model(batch)
          predicted_labels = (torch.sigmoid(logits) > 0.5).long()
          if not batch_idx % 100:
            print ('Epoch: {',batch_idx)
          for p,a in zip(predicted_labels,batch['targets'].to(device).long()): 
            predicted_label=int(p.item())
            actual_label=int(a.item())
            if (predicted_label==1 and actual_label==1):
              tp += 1
            if (predicted_label==0 and actual_label==0):
              tn += 1
            if (predicted_label==0 and actual_label==1):
              fn += 1
            if (predicted_label==1 and actual_label==0):
              fp += 1
             
            #fp += (predicted_labels == batch_data[1].to(device).long()).sum()
        print("tp,tn,fn,fp is : ",tp,tn,fn,fp)
        if (tp>0 ):
          metrics=dict()    
          metrics["precision"]=(tp/(tp+fp))
          metrics["recall"]=(tp/(tp+fn))
          metrics["f1_score"]=(2*metrics["precision"]*metrics["recall"])/(metrics["precision"]+metrics["recall"])
          print("F1 score is ",metrics["f1_score"] )
        else:
          metrics=dict()   
          print("F1 score is 0")
          metrics["precision"]=0
          metrics["recall"]=0
          metrics["f1_score"]=0
            
    return metrics
class Trainer:
    def __init__(self, modelName,model,train_partial_dataset, val_dataset, loss_function, optimizer,path_save_best_model,device):
        
        self.train_dataset = train_dataset 
        self.test_dataset = val_dataset
        self.model = model
        self.modelName=modelName
        self.loss = loss_function 
        self.optimizer = optimizer
        self.device = torch.device( device if torch.cuda.is_available() else 'cpu')
        self.model.to( self.device )
        print('using device: ',self.device)

    def _train(self, train_dataset, num_epochs=1,clip=True ):

        train_dataloader = DataLoader( self.train_dataset, batch_size= batch_size, shuffle= False, num_workers= num_dataset_workers)
        val_dataloader = DataLoader( self.test_dataset, batch_size= batch_size, shuffle= False, num_workers= num_dataset_workers)
        #test_dataloader = DataLoader( test_dataset, batch_size= batch_size, shuffle= False, num_workers= num_dataset_workers)
        best_val_f1= -1
        current_best = -1
        best_res = None
        
        torch.cuda.empty_cache()
        start_time = time.time()
        for epoch in range(num_epochs):
          self.model.train()
          train_losses=[]
          print("epoch is ",epoch)
          for batch_idx, batch in enumerate(train_dataloader):
            batch = { k : v.to( self.device ) for k,v in batch.items() }
            logits,_ = model(batch)
            #resize=batch_dat_function[0].size(0)
            cost = self.loss(logits, batch['targets'].float().to(self.device))
            optimizer.zero_grad()
            cost.backward()
            if clip:
                torch.nn.utils.clip_grad_norm(model.parameters(),0.5)
            optimizer.step()
            train_losses.append(cost.item())
          
          print('train loss is ',np.average(train_losses))
          with torch.set_grad_enabled(False):
            matrix=compute_binary_accuracy(model, val_dataloader, DEVICE)
            curr_val_f1=matrix["f1_score"]
            curr_val_precision=matrix["precision"]
            curr_val_recall=matrix["recall"]
          if curr_val_f1>best_val_f1 :
            best_val_f1=curr_val_f1
            
            print("Saving Best Matrix",matrix)
            counter=0
            torch.save( self.model.state_dict(), path_save_best_model)
            #dill.dump(self.model, open(BestModelDir+ "/"+modelName+"classifier.pkl","wb"))
            SaveBestResult(matrix,ResultfilePath,self.modelName)
          else :
            print("F1-Score has not improved and it is : ",best_val_f1)
          print('Time elapsed: {',(time.time() - start_time)/60 )

class Result:
    def __init__( self, review_id, review_text, actual_label,predicted_label ):
            
        self.review_id = review_id
        self.review_text = review_text
        self.actual_label = actual_label
        self.predicted_label=predicted_label
        
        
        
    
    def __str__(self):
        return str(self.__dict__)

def result_dataset_generator(test_dataloader):
  
  p=0
  result_dataset_list=[]
  #print('ID','Sentence','Actual Label','Predicted Label')
  for batch_idx, batch in enumerate(val_dataloader):
    batch = { k : v.to( device ) for k,v in batch.items() } 
    actual_label=batch['targets'][0].tolist()
    logits,att = model(batch)
    predicted_label = (torch.sigmoid(logits) > 0.5).long()
    predicted_label=predicted_label[0].tolist()
    id=batch['id'][0].tolist()
    Input_data=batch['review'][0].tolist()
    Input_actual_length=batch['original_review_length'][0].tolist()
    Input_data=Input_data[:Input_actual_length]
    sentence = ' '.join([vocab.id_to_word[word] for word in Input_data])
    #p+=1
    result_dataset_list.append(Result(id,sentence,actual_label,predicted_label))
  return  result_dataset_list

def result_writer(output_file_path,result_list):
  with open( output_file_path, 'w' ) as f:
    f.write('id'+ '\t' +'comment'+ '\t' +'actual_label'+ '\t'+'predicted_label\n')
    for i in result_list:
      f.write( str(i.review_id) + '\t' + str(i.review_text) + '\t'+str(i.actual_label) + '\t'+str(i.predicted_label))
      f.write('\n')
    f.close
	
def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, str(word))	
	
def mk_html(sentence, attns):
    html = ""
    for word, attn in zip(sentence, attns):
        #print(vocab.id_to_word[word])
        #print(attn)
        html += ' ' + highlight(
            vocab.id_to_word[word],
            attn
        )
    return html + "<br><br>\n"+ "<br><br>\n"+ "<br><br>\n"
	
if __name__ == "__main__":

    train_dataset = ReviewDataset(Clean_Train_Data, preprocessed= True, vocab= vocab)
    test_dataset = ReviewDataset(Clean_Test_Data, preprocessed= True, vocab= vocab)
    val_dataloader = DataLoader( test_dataset, batch_size= 1, shuffle= False, num_workers= 20)                                                 
    batch_size=128
    num_dataset_workers=20
    LEARNING_RATE=0.0005
    embedding_length=100
    hidden_size=100
    output_size=1
    modelName="Unbal_MultiHeadAttBiLSTM"
    
    model = MultiHeadAttBiLSTM(weights, embedding_length, hidden_size, output_size,batch_size)
    path_save_best_model=os.path.join(BestModelDir+ "/"+modelName+".pt")
                                      
    loss_function=F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    trainer = Trainer(modelName,model, train_dataset, test_dataset,loss_function, optimizer,path_save_best_model,DEVICE)
    trainer._train(train_dataset)
    result_list=result_dataset_generator(val_dataloader)
    p=0
    p1=0
    p2=0
    p3=0
    p4=0
    f1 = open(rootDir+"TP1_attn.html", "w")
    for batch_idx, batch in enumerate(val_dataloader):
      batch = { k : v.to( device ) for k,v in batch.items() } 
      actual_label=batch['targets'][0].tolist()
      logits,att = model(batch)
      predicted_label = (torch.sigmoid(logits) > 0.5).long()
      #print(att.shape)
      #print(att)
      p+=1
      whole_att = att.data[0,0,:].tolist()
      Input_data=batch['review'][0].tolist()
      Input_actual_length=batch['original_review_length'][0].tolist()
      actual_input_data=Input_data[:Input_actual_length]
      actual_att=whole_att[:Input_actual_length]
      #print (actual_label,predicted_label[0].tolist())
      if actual_label == predicted_label[0].tolist() and actual_label ==1  and max(actual_att) > 0.2:
        #f1.write('\t'.join(( str(actual_label),str(predicted_label[0].tolist()),mk_html(actual_input_data, actual_att))))
        f1.write('\t'.join(( str(' '),str(' '),mk_html(actual_input_data, actual_att))))
        print("max attention value is ",max(actual_att))
        print("\n")
        p1+=1
  
      if p%10000==0:
        print("p1,",p1)
      if p1==1000  :
        f1.close()
        break;	