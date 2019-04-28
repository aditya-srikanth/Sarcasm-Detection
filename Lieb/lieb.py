# SVM This is a reimplementation of 'The perfect solution for detecting sarcasm in tweets# not' Lieberecht et al (2013)
#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import pickle
import re
from scipy.sparse import csr_matrix, lil_matrix
from scipy import io

dataset_path = '../data/balanced_test.tsv'
out_path = 'lieb_balanced_test.mtx'

f = open(dataset_path, 'r', encoding="utf-8-sig")
qid = 0
dict = {}
word_count = {}
rev_dict = {}
index = 0
row = np.zeros(())


def bigramgen(input):
    input = str(input.strip())
    input = input.lower()
    words = input.split(' ')
    prev_word = 'init'
    output = ''
    for word in words:
        output += prev_word+word+' '
        prev_word = word

    prev_word = 'init'
    prev_prev_word = 'init'
    for word in words:
        output += prev_prev_word+prev_word+word+' '
        prev_prev_word = prev_word
        prev_word = word
    output += input.strip()
    return output


line_count = 0

# main method starts
for line in f:
    line_count += 1
    contents = line.split('\t')
    if len(contents) == 2 and "Scene" not in line:
        dialogue = contents[0].lower()

        if len(dialogue) == 0:
            continue

        words = re.findall(r"[\w']+|[.:,!?;]", dialogue)
        # first_word = words[0]

        stitched = ''
        for word in words:
            stitched += word+' '

        stitched = stitched.strip()
        stitched = bigramgen(stitched)
        words = stitched.split(' ')
        for word in words:
            if word not in dict:
                dict[word] = index
                rev_dict[index] = word
                index += 1
                word_count[word] = 1
            else:
                word_count[word] += word_count[word]

def gen_features(dataset_path,out_path):
    global index,qid
    f = open(dataset_path, 'r', encoding='utf-8-sig')
    line_count = len(list(f))
    print(line_count)
    f = open(dataset_path, 'r', encoding='utf-8-sig')
    f_o1 = open('out', 'w')
    f_o1.write('# Vocabulary size:'+str(index)+'\n')
    # print(len(dict))
    # print(dict)
    print(len(dict))
    data = lil_matrix((line_count, len(dict)))
    line_count = 0
    for line in f:
        print(line_count)
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
            #dialogue = dialogue + ' '+ getActions(dialogue).lower()
            if len(dialogue) == 0:
                continue
            words = re.findall(r"[\w']+|[.,!?;]", dialogue)

            # first_word = words[0]
            # speaker = first_word+':'

            stitched = ''
            for word in words:
                stitched += word+' '

            stitched = stitched.strip()
            stitched = bigramgen(stitched)
            words = stitched.split(' ')

            for word in words:
                if word in dict:

                    index = dict[word]
                    if word_count[word] >= 3:
                        word_ids.append(index)

            word_ids = list(set(word_ids))
            word_ids.sort()

            s_line = ''
            # print(word_ids)
            for id in word_ids:
                s_line += str(id)+':1 '
                data[line_count, id] = 1
            # s_line += '# ' + stitched
            s_line = s_line.strip()
            f_o1.write(s_line+'\n')
            line_count += 1

    # data = pd.DataFrame(data)
    data = csr_matrix(data)
    # with open("lieb.pkl", 'wb') as f:
    #     pickle.dump(data, f)
    print(data.shape)
    io.mmwrite(out_path, data)
    # print(dict)
    print(('gravity' in dict))
    # print(dict)


dataset_path = '../data/balanced_train.tsv'
out_path = 'lieb_balanced_train.mtx'
gen_features(dataset_path,out_path)

dataset_path = '../data/balanced_test.tsv'
out_path = 'lieb_balanced_test.mtx'
gen_features(dataset_path,out_path)