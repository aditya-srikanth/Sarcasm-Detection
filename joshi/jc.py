import re
from scipy.sparse import csr_matrix, lil_matrix
from scipy import io

dataset_path = '../data/unbalanced_train.tsv'
test_path = '../data/unbalanced_test.tsv'
out_path = 'jc_unbal_train.mtx'
out_path_test = 'jc_unbal_test.mtx'


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


def getExplicit(input, i_base):
    output = []
    input = str(input).lower()
    words = re.findall(r"[\w']+|[.:,!?;]", input)
    abs_pos_score = 0
    flips = 0
    largest_sequence = 0
    curr_sequence = 0
    abs_neg_score = 0
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
        output.append((id, word_count[id]))

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
            s_explicit = getExplicit(line, i_base)
            s_implicit = getImplicitFeatures(line)

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
            feat.append(s_explicit)
            feat.append(s_implicit)

            for feature in feat:
                if feature == None:
                    continue
                elif type(feature) != list:
                    data[line_index, (feature[0])-1] = feature[1]
                if type(feature) == list:
                    for feature_element in feature:
                        data[line_index, (feature_element[0]) -
                             1] = feature_element[1]

            line_index += 1
            print(line_index)

    # Save it as mtx
    print('Final Data:'+str(data.shape))
    io.mmwrite(out_path, data)
    print('Finished Saving')


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

# Add Implicit Dictionary also
for phrase in implicit_arr:
    implicit_dict[phrase] = index
    index += 1

# Joshi Features
i_excl = index+1
i_quest = index+2
i_dotdot = index + 3
i_interj = index + 4
i_base = index + 5
final_features = i_base + 5

# Main Extraction of Features
getFeatures(dataset_path, out_path)
getFeatures(test_path, out_path_test)
