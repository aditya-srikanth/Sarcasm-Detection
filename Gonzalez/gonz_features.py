import re
from scipy.sparse import lil_matrix, csr_matrix
from scipy import io
import numpy as np

dataset_path = '../data/unbalanced_train.tsv'
test_path = '../data/unbalanced_test.tsv'
out_path = 'gonz_unbal_train.mtx'
out_path_test = 'gonz_unbal_test.mtx'


def getLIWCFeatures(input, i_base):

    input = str(input)
    input = input.lower()

    liwc_dict = {'LP': 0, 'PP': 0, 'PC': 0}
    words = input.split(' ')

    for word in words:
        app_features = ''
        for key in sentiment_dict.keys():
            if word.startswith(key):
                app_features = sentiment_dict[key]

        if app_features.lower() == '':
            continue

        for app_feature in app_features.split(' '):
            if app_feature in featuremap_dict:
                category = featuremap_dict[app_feature]
                liwc_dict[category] += 1

    output = []
    output.append((i_base, liwc_dict['LP']))
    output.append((i_base+1, liwc_dict['PP']))
    output.append((i_base+2, liwc_dict['PC']))

    return output


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


def getFeatures(dataset_path, out_path):
    f = open(dataset_path, 'r', encoding='utf-8-sig')
    num_rows = len(list(f))
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
            s_liwc = getLIWCFeatures(dialogue, i_liwcbase)

            feat = []
            word_ids = []
            words = re.findall(r"[\w']+|[.,!?;]", dialogue)
            for word in words:
                if word in dict:
                    index = dict[word]
                    if word_count[word] >= 3:
                        word_ids.append(index)

            word_ids = list(set(word_ids))
            word_ids.sort()
            # Unigrams
            for id in word_ids:
                feat.append((id, 1))
            feat.append(s_punct)
            feat.append(s_interj)
            feat.append(s_liwc)

            # Add it to data
            for feature in feat:
                if feature == None:
                    continue
                elif type(feature) != list:
                    # print(feature)
                    data[line_index, (feature[0])-1] = feature[1]
                if type(feature) == list:
                    for feature_element in feature:
                        data[line_index, (feature_element[0]) -
                             1] = feature_element[1]

            line_index += 1
            print(line_index)

    # Save it as mtx
    data = csr_matrix(data)
    print('Final Data:'+str(data.shape))
    io.mmwrite(out_path, data)
    print('Finished Saving')


# LIWC DICTIONARY
featuremap_dict = {'16': 'LP', '12': 'LP', '19': 'LP', '141': 'LP', '142': 'LP', '143': 'LP', '146': 'LP', '22': 'PP', '125': 'PP',
                   '126': 'PP', '127': 'PP', '128': 'PP', '129': 'PP', '130': 'PP', '131': 'PP', '132': 'PP', '133': 'PP', '134': 'PP',
                   '135': 'PP', '136': 'PP', '137': 'PP', '138': 'PP', '139': 'PP', '140': 'PP', '366': 'PP', '121': 'PC', '122': 'PC', '123': 'PC',
                   '124': 'PC', '148': 'PC', '149': 'PC', '150': 'PC', '354': 'PC', '356': 'PC', '357': 'PC', '358': 'PC', '359': 'PC'}

# PREPARE DICTIONARY WITH MAPPINGS FROM WORD TO APP_FEATURE
f = open('LIWC_words', 'r')
sentiment_dict = {}

for line in f:
    words = line.split('\t')
    temp_string = ''
    for i in range(1, len(words)):
        temp_string += words[i]+' '
    sentiment_dict[words[0].replace('*', '')] = temp_string.strip()

# PREPARE INTERJECTIONS ARRAY
f = open('interj_words', 'r')
interj_arr = []

for line in f:
    words = line.split(' ')

    if len(words) > 0:
        interj_arr.append(words[0].strip())

# PREPARE UNIGRAMS DICTIONARY
num_examples = 0
f = open(dataset_path, 'r', encoding='utf-8-sig')

dict = {}
word_count = {}
rev_dict = {}
index = 1

for line in f:
    num_examples += 1
    contents = line.split('\t')
    if len(contents) == 2 and "Scene" not in line:
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

# Gonazalez Featuers
i_excl = index+1
i_quest = index+2
i_dotdot = index+3
i_interj = index+4
i_liwcbase = index+5
final_features = i_liwcbase + 2

# Initialize Resulting Feature Vectors
getFeatures(dataset_path, out_path)
getFeatures(test_path, out_path_test)
