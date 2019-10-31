import re
from scipy.sparse import csr_matrix, lil_matrix
from scipy import io
import pandas as pd
import sys

dataset_path = '../ddata/balanced_train.tsv'
test_path = '../ddata/balanced_test.tsv'
out_path = 'jc_balanced_train.mtx'
out_path_test = 'jc_balanced_test.mtx'

# dataset_path = sys.argv[1]
# test_path = sys.argv[3]
# out_path = sys.argv[2]
# out_path_test = sys.argv[4]


def getInterjection(input, i_interj):
    count = 0
    input = input.lower()

    for i in range(0, len(interj_arr)):
        count += input.count(interj_arr[i])

    return (i_interj, count)


def getPunctuation(input, i_excl, i_quest, i_dotdot):
    output = []
    output.append((i_excl, input.count('!')))
    output.append((i_quest, input.count('?')))
    output.append((i_dotdot, input.count('...')))
    return output


def getExplicit(input_text, i_base):
    output = []
    input_text = str(input_text).lower()
    words = re.findall(r"[\w']+|[.:,!?;]", input_text)
    abs_pos_score = 0
    flips = 0
    largest_sequence = 0
    curr_sequence = 0
    abs_neg_score = 0
    last_polarity = 1

    for word in words:
        if word.lower() in sentiment_dict:
            sentiment = sentiment_dict[word.lower()]

            if int(sentiment) == 1:
                print(word+' found as positive')
                abs_pos_score += 1
                if last_polarity == 1:
                    curr_sequence += 1
                else:
                    flips += 1
                    if curr_sequence > largest_sequence:
                        largest_sequence = curr_sequence
                    curr_sequence = 0
                last_polarity = 1

            else:
                print(word+' found as negative')
                abs_neg_score += 1
                if last_polarity == -1:
                    curr_sequence += 1
                else:
                    flips += 1
                    if curr_sequence > largest_sequence:
                        largest_sequence = curr_sequence
                    curr_sequence = 0
                last_polarity = -1
    # print(abs_pos_score)
    # print(abs_neg_score)

    if abs_pos_score > abs_neg_score:
        polarity = 1
    elif abs_neg_score > abs_pos_score:
        polarity = 2
    else:
        polarity = 3
    # print(polarity)
    # print("-----")
    output.append((i_base, abs_pos_score))
    output.append((i_base+1, abs_neg_score))
    output.append((i_base+2, polarity))
    output.append((i_base+3, largest_sequence))
    output.append((i_base+4, flips))
    print(input_text)
    print('largest_sequence', largest_sequence)
    input()
    return output


def getImplicitFeatures(inp):
    output = []
    inp = str(inp).lower()

    word_count = {}
    word_ids = []
    words = re.findall(r"[\w']+|[.:,!?;]", inp)
    # print(words)
    for word in words:
        if word in implicit_dict:
            # print(str(word)+": in implicit_dict")
            index = implicit_dict[word]
            # print(str(index) + ": index")
            if index in word_count:
                word_count[index] += 1
            else:
                word_count[index] = 1
                # print(word)
                # print(index)
                word_ids.append(index)

    word_ids = list(set(word_ids))
    word_ids.sort()

    for id in word_ids:
        output.append((id, word_count[id]))

    return output


def getFeatures(dataset_path, out_path):
    f = open(dataset_path, 'r', encoding='utf-8-sig')
    num_rows = len(list(f))
    print(num_rows)
    input()
    data = lil_matrix((num_rows, final_features))
    f = open(dataset_path, 'r', encoding='utf-8-sig')
    line_index = 0
    for line in f:
        contents = line.split('\t')

        if len(contents) >= 2:
            dialogue = contents[0].lower()

            if len(dialogue) == 0:
                continue

            s_punct = getPunctuation(line, i_excl, i_quest, i_dotdot)
            s_interj = getInterjection(line, i_interj)
            s_explicit = getExplicit(line, i_base)
            s_implicit = getImplicitFeatures(line)

            feat = []
            word_ids = []
            words = re.findall(r"[\w']+|[.,!?;]", dialogue)
            for word in words:
                if word in dict:
                    index = dict[word]
                    if word_count[word] >= 0:
                        word_ids.append(index)

            word_ids = list(set(word_ids))
            word_ids.sort()

            # Unigrams
            for id in word_ids:
                feat.append((id, 1))
            feat.append(s_punct)
            feat.append(s_interj)
            feat.append(s_explicit)
            feat.append(s_implicit)

            for feature in feat:
                if feature == None:
                    continue
                elif type(feature) != list:
                    data[line_index, (feature[0])] = feature[1]
                if type(feature) == list:
                    for feature_element in feature:
                        data[line_index, (feature_element[0])
                             ] = feature_element[1]

            line_index += 1
            print(line_index)

    # Save it as mtx
    print('Final Data:'+str(data.shape))
    io.mmwrite(out_path, data)
    print('Finished Saving')
#     ans = data.todense()
    ans = 0
    return ans


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

# implicit_arr = list(set(implicit_arr))

# PREPARE UNIGRAMS DICTIONARY
num_examples = 0
f = open(dataset_path, 'r', encoding='utf-8-sig')

dict = {}
word_count = {}
rev_dict = {}
index = 0

for line in f:
    num_examples += 1
    contents = line.split('\t')
    if len(contents) >= 2 and "Scene" not in line:
        dialogue = contents[0].lower()
        if len(dialogue) == 0:
            continue

        words = re.findall(r"[\w']+|[.:,!?;]", dialogue)

        for word in words:
            if word not in dict:
                dict[word] = index
                rev_dict[index] = word
                index += 1
                word_count[word] = 1
            else:
                word_count[word] += word_count[word]

# Print
names = []
for key in rev_dict:
    names.append(rev_dict[key])
# print(names)

unig = index
print(unig)

# Add Implicit Dictionary also
for phrase in implicit_arr:
    if phrase in implicit_dict:
        continue
    implicit_dict[phrase] = index
    index += 1

for key in implicit_dict:
    names.append(key)
# print(names)

# Joshi Features
i_excl = index
i_quest = index+1
i_dotdot = index + 2
i_interj = index + 3
i_base = index + 4
final_features = i_base + 5

names.append('Excalamation')
names.append('Question_Mark')
names.append('dotdotdot')
names.append('Interjection')
names.append('Abs_Pos_Score')
names.append('Abs_Neg_Score')
names.append('Polarity')
names.append('Largest_Sequence')
names.append('Flips')

# Main Extraction of Features
# ans = getFeatures(dataset_path, out_path)
ans2 = getFeatures(test_path, out_path_test)
# df = pd.DataFrame(ans, columns=names)
# df.to_csv('joshi.csv')
print(index-unig)
