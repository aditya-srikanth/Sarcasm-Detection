# SVM: This is a reimplementation of 'Harnessing context incongruity' by Joshi et al (2015)
#!/usr/bin/python

import sys
import re
from scipy.sparse import csr_matrix, lil_matrix
from scipy import io

dataset_path = '../new_data_unbalanced.tsv'
out_path = 'jco_unbalanced_csr.mtx'

# Sentiment wordlist load
f = open('sentiwordlist', 'r')
sentiment_dict = {}

for line in f:
    words = line.split(' ')
    sentiment_dict[words[0]] = words[1]

# Interjection wordlist load
f = open('interj_words', 'r')
interj_arr = []

for line in f:
    words = line.split(' ')

    if len(words) > 0:
        interj_arr.append(words[0].strip())

# implicit phrases
f = open('implicit_phrases', 'r')
implicit_arr = []
implicit_dict = {}

for line in f:
    words = line.split(' ')

    if len(words) > 0:
        implicit_arr.append(words[0].strip())

# explicit flip features


def getExplicit(input, i_base):
    output = []
    input = str(input).lower()
    words = re.findall(r"[\w']+|[.:,!?;]", input)
    abs_pos_score = 0
    flips = 0
    largest_sequence = 0
    curr_sequence = 0
    abs_neg_score = 0
    pos_score = 0
    neg_score = 0
    last_polarity = +1

    for word in words:
        if word.lower() in sentiment_dict:
            sentiment = sentiment_dict[word.lower()]

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

    output.append((i_base, abs_pos_score))
    output.append((i_base+1, abs_neg_score))
    output.append((i_base+2, polarity))
    output.append((i_base+3, largest_sequence))
    output.append((i_base+4, flips))

    return output


# Return count of interjections
def getInterjection(input, i_interj):
    output = ''
    count = 0
    input = str(input).lower()
    for i in range(0, len(interj_arr)):
        count += input.count(interj_arr[i])

    return (i_interj, count)


f = open(dataset_path, 'r')
qid = 0
dict = {}
word_count = {}
rev_dict = {}
index = 1

# punctuations


def getPunctuation(input, i_excl, i_quest, i_dotdot):
    output = []
    output.append((i_excl, input.count('!')))
    output.append((i_quest, input.count('?')))
    output.append((i_dotdot, input.count('...')))
    return output

# implicit phrase features


def getImplicitFeatures(input):
    output = []
    input = str(input).lower()

    word_count = {}
    word_ids = []
    words = re.findall(r"[\w']+|[.:,!?;]", input)

    for word in words:
        if word in implicit_dict:
            index = implicit_dict[word]
            if index in word_count:
                word_count[index] += 1
            else:
                word_count[index] = 1
                word_ids.append(index)

    word_ids = list(set(word_ids))
    word_ids.sort()

    for id in word_ids:
        # output += str(id)+':'+str(word_count[id])+' '
        output.append((id, word_count[id]))
    # output = output.strip()

    return output


# main method starts
num_examples = 0
for line in f:
    contents = line.split('\t')
    num_examples += 1
    if len(contents) == 2 and "Scene" not in line:
        dialogue = contents[0].lower()

        if len(dialogue) == 0:
            continue

        words = re.findall(r"[\w']+|[.:,!?;]", dialogue)
        first_word = words[0]

        for word in words:
            if word not in dict:
                dict[word] = index
                rev_dict[index] = word
                index += 1
                word_count[word] = 1
            else:
                word_count[word] += word_count[word]

print(index)

for phrase in implicit_arr:
    implicit_dict[phrase] = index
    index += 1

print(index)

i_excl = index+1
i_quest = index+2
i_dotdot = index + 3
i_interj = index + 4
i_base = index + 5
final_features = i_base + 5

row = csr_matrix((1, final_features))
data = lil_matrix((num_examples, final_features))

f = open(dataset_path, 'r')
f_o1 = open('out.txt', 'w')
f_o1.write('# Vocabulary size:'+str(index)+'\n')
line_index = 0
for line in f:
    print(line_index)
    s_line = ''
    contents = line.split('\t')
    pos_score = 0
    neg_score = 0
    if "Scene" in line:
        qid += 1

    if len(contents) >= 2:
        # print(contents)
        word_ids = [1]
        dialogue = contents[0].lower()

        if len(dialogue) == 0:
            continue
        words = re.findall(r"[\w']+|[.,!?;]", dialogue)

        first_word = words[0]
        speaker = first_word+':'
        words.append(speaker)

        s_punct = getPunctuation(line, i_excl, i_quest, i_dotdot)
        s_interj = getInterjection(line, i_interj)
        s_explicit = getExplicit(line, i_base)
        s_implicit = getImplicitFeatures(line)

        for word in words:
            if word in dict:

                index = dict[word]
                if word_count[word] >= 3:
                    word_ids.append(index)

        # if contents[1].strip().lower() == 'sarcasm':
        #     label = '+1'
        # else:
        #     label = '-1'
        line = []
        word_ids = list(set(word_ids))
        word_ids.sort()
        # s_line = label+' '
        # print(word_ids)
        for id in word_ids:
            s_line += str(id)+':1 '
            line.append((id, 1))
        if s_implicit != None:
            line.append(s_implicit)
        if s_punct != None:
            line.append(s_punct)
        line.append(s_interj)
        line.append(s_explicit)

        for feature in line:
            if feature == None:
                continue
            elif type(feature) != list:
                # print(feature)
                data[line_index, (feature[0])-1] = feature[1]
            if type(feature) == list:
                for feature_element in feature:
                    data[line_index, (feature_element[0]) -
                         1] = feature_element[1]

        s_line += str(s_implicit)+' '+str(s_punct)+' ' + \
            str(s_interj)+' '+str(s_explicit)

        # s_line += ' # '+line
        line_index += 1
        s_line = s_line.strip()
        f_o1.write(s_line+'\n')

data = data.tocsr()
print(data.shape)
io.mmwrite(out_path, data)
