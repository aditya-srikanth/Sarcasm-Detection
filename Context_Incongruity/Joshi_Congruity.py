import sys
import re 
import pandas as pd
import numpy as np

sentiment_word_path = 'sentiwordlist'
interjection_path = 'interj_words'
implicit_path = 'implicit_phrases'

class JoshiCongruity:

    def __init__(self):
        self.feat_df = pd.DataFrame()
        self.feat_df_n = pd.DataFrame()
        pass
    
    def initDict(self):
        self.sentiment_dict = {}
        self.interj_arr = []
        self.implicit_arr = []
        self.implicit_dict = {}

        f = open(sentiment_word_path, 'r', encoding= 'utf-8-sig')
        for line in f:
            words = line.split(' ')
            self.sentiment_dict[words[0]] = words[1]

        f = open(interjection_path, 'r', encoding= 'utf-8-sig')
        for line in f:
            words = line.split(' ')
            if len(words) > 0:
                self.interj_arr.append(words[0].strip())
        
        f = open(implicit_path, 'r', encoding= 'utf-8-sig')
        for line in f:
            words = line.split(' ')
            if len(words) > 0:
                self.implicit_arr.append(words[0].strip())
    
    def getExplicit(self, input, i_base):
        output = ''
        input = str(input).lower()
        words = re.findall(r"[\w]+|[.:,!?;]",input)
        abs_pos_score = 0
        flips = 0
        largest_sequence = 0
        curr_sequence = 0
        abs_neg_score = 0
        last_polarity = +1

        for word in words:
            if word.lower() in self.sentiment_dict:
                        sentiment = self.sentiment_dict[word.lower()]

                        if int(sentiment) == 1:
                            #print(word+' found as positive')
                            abs_pos_score += 1
                            if last_polarity == 1:
                                curr_sequence += 1
                            else:
                                flips += 1
                                if largest_sequence > curr_sequence:
                                    largest_sequence = curr_sequence
                                curr_sequence = 0
                            last_polarity = 1						

                        else:
                            #print(word+' found as negative')
                            abs_neg_score += 1
                            if last_polarity == -1:
                                curr_sequence += 1
                            else:
                                flips += 1
                                if largest_sequence > curr_sequence:
                                    largest_sequence = curr_sequence
                                curr_sequence = 0
                            last_polarity = -1



        if (abs_pos_score > abs_neg_score):
            polarity = 1
        elif (abs_neg_score > abs_pos_score):
            polarity = 2
        else:
            polarity = 3 	

        output += str(i_base)+':'+str(abs_pos_score)+' '
        self.row[i_base] = abs_pos_score
        # self.feat_df.loc[self.i, str(i_base)] = str(abs_pos_score) 
        output += str(i_base+1)+':'+str(abs_neg_score)+' '
        self.row[i_base+1] = abs_neg_score
        # self.feat_df.loc[self.i, str(i_base+1)] = str(abs_neg_score) 
        output += str(i_base+2)+':'+str(polarity)+' '
        self.row[i_base+2] = polarity
        # self.feat_df.loc[self.i, str(i_base+2)] = str(polarity) 
        output += str(i_base+3)+':'+str(largest_sequence)+' '
        self.row[i_base+3] = largest_sequence
        # self.feat_df.loc[self.i, str(i_base+3)] = str(largest_sequence) 
        output += str(i_base+4)+':'+str(flips)
        self.row[i_base+4] = flips
        # self.feat_df.loc[self.i, str(i_base+4)] = str(flips) 

        return output.strip()
    
    def getInterjection(self, input, i_interj):
        output = ''
        count = 0
        input = str(input).lower()
        for i in range(0,len(self.interj_arr)):
            count += input.count(self.interj_arr[i])

        output = str(i_interj)+':'+str(count)
        self.row[i_interj] = count
        # self.feat_df.loc[self.i, str(i_interj)] = str(count) 
        return output
    
    def getPunctuation(self, input, i_excl, i_quest, i_dotdot):
        output = ''
        input = str(input).strip()
        output += str(i_excl)+':'+str(input.count('!')) +' '
        self.row[i_excl] = input.count('!')
        # self.feat_df.loc[self.i, str(i_excl)] = str(input.count('?!'))
        output += str(i_quest)+':'+str(input.count('?'))+' '
        self.row[i_quest] = input.count('?')
        # self.feat_df.loc[self.i, str(i_quest)] = str(input.count('?'))
        output += str(i_dotdot)+':'+str(input.count('...'))
        self.row[i_dotdot] = input.count('...')
        # self.feat_df.loc[self.i, str(i_dotdot)] = str(input.count('...'))
        return output.strip()

    def getImplicitFeatures(self, input):
        output = ''
        input = str(input).lower()

        word_count = {}
        word_ids = []
        words = re.findall(r"[\w]+|[.:,!?;]",input)

        for word in words:
            if word in self.implicit_dict:
                index = self.implicit_dict[word]
                if index in word_count:
                    word_count[index] += 1
                else:
                    word_count[index] = 1
                    word_ids.append(index)

        word_ids = list(set(word_ids))
        word_ids.sort()

        for id in word_ids:
            output += str(id)+':'+str(word_count[id])+' '
            self.row[id] = word_count[id]
            # self.feat_df.loc[self.i, str(id)] = str(word_count[id])
        output = output.strip()

        return output

    def extractFeatures(self, data_path, out_path):
        self.initDict()
        f = open(data_path, 'r', encoding='utf-8-sig')
        qid = 0
        dict = {}
        word_count = {}
        rev_dict = {}
        index = 1
        for line in f:
            contents = line.split('\t')
            if len(contents) == 2 and 'Scene' not in line:
                dialogue = contents[0].lower()
                
                if len(dialogue) == 0:
                    continue
                words = re.findall(r"[\w]+|[.:,!?;]",dialogue)
                first_word = words[0]

                for word in words:
                    if word not in dict:
                        dict[word] = index
                        rev_dict[index] = word
                        index += 1
                        word_count[word] = 1
                    else:
                        word_count[word] += word_count[word]
        
        for phrase in self.implicit_arr:
            self.implicit_dict[phrase] = index
            index += 1 
        
        i_excl      = index + 1
        i_quest     = index + 2
        i_dotdot    = index + 3
        i_interj    = index + 4
        i_base      = index + 5

        f = open(data_path, 'r', encoding='utf-8-sig')
        f_o1 = open(out_path, 'w', encoding='utf-8-sig')

        self.i = 0
        self.mat = np.zeros((3630, i_base+5), dtype='int')
        for line in f:
            
            self.row = np.zeros(i_base+5)
            contents = line.split('\t')
            s_line = ''

            if "Scene" in line:
                qid +=1

            if len(contents) >=2:
                word_ids = [1]
                dialogue = contents[0].lower()

                if len(dialogue) == 0:
                    continue
                words = re.findall(r"[\w']+|[.,!?;]",dialogue)

                first_word = words[0]
                speaker = first_word+':'
                words.append(speaker)


                s_punct = self.getPunctuation(line,i_excl,i_quest,i_dotdot)
                s_interj = self.getInterjection(line,i_interj)
                s_explicit = self.getExplicit(line,i_base)
                s_implicit = self.getImplicitFeatures(line)

                for word in words:
                    if word in dict:
                        index = dict[word]
                        if word_count[word] >= 3:
                            word_ids.append(index)

                if contents[1].strip().lower() == 'sarcasm':
                    label = '+1'
                else:
                    label = '-1'

                word_ids = list(set(word_ids))
                word_ids.sort()
                s_line = label+' '
                for id in word_ids:
                    s_line += str(id)+':1 '
                    self.row[id] = 1

                s_line += s_implicit+' '+s_punct+' '+s_interj+' '+s_explicit
                # s_line += ' # '+line
                s_line = s_line.strip()
                f_o1.write(s_line+'\n')
                
                
                self.mat[self.i] = self.row
                self.i += 1
            
        
        self.feat_df = pd.DataFrame(self.mat)

cong = JoshiCongruity()
cong.extractFeatures('./dataset', 'jc_features')

df = cong.feat_df
df = df.loc[:, 1:]
df = df.loc[0:3628 , :]
df.to_pickle('jc_features_df.pkl')
# cong.feat_df.to_csv('df.csv')