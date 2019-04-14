import pickle
import pandas as pd
import numpy as np
import re


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


# Return true if quotation marks are available
def getQuotes(input, i_quotes):
    output = ''
    if '"' in str(input):

        return (i_quotes, 1)
    else:
        return (i_quotes, 0)


def getHyperbole(input, i_hyp):
    output = ''
    input = str(input).lower()
    words = re.findall(r"[\w']+|[.:,!?;]", input)

    pos_score = 0
    neg_score = 0
    for word in words:
        if word.lower() in sentiment_dict:
            sentiment = sentiment_dict[word.lower()]

            if int(sentiment) == 1:
                #print(word+' found as positive')
                pos_score += 1
                neg_score = 0
            else:
                #print(word+' found as negative')
                neg_score += 1
                pos_score = 0
        else:
            pos_score = 0
            neg_score = 0
        global final_features
        row = np.zeros((1, final_features))
        if pos_score == 3:
            row[i_hyp] = pos_score
            return (i_hyp, pos_score)
        elif neg_score == 3:
            return (i_hyp, neg_score)

    return (i_hyp, 0)

# Return count of interjections


def getInterjection(input, i_interj):
    output = ''
    count = 0
    input = str(input).lower()
    for i in range(0, len(interj_arr)):
        count += input.count(interj_arr[i])

    return (i_interj, count)


def getPosNegPunct(input, i_pnpunct):
    output = ''
    count = 0
    input = str(input).lower()
    words = input.split(' ')
    words = re.findall(r"[\w']+|[.:,!?;]", input)
    # TODO: handle this case!!!!
    if '!' not in input and '?' not in input:
        return None

    pos_score = 0
    neg_score = 0

    for word in words:
        if word.lower() in sentiment_dict:
            sentiment = sentiment_dict[word.lower()]

            if int(sentiment) == 1:
                #print(word+' found as positive')
                pos_score += 1
            else:
                #print(word+' found as negative')
                neg_score += 1

        if pos_score >= 1 and neg_score == 0:
            return (i_pnpunct, 1)
        elif pos_score == 0 and neg_score >= 1:
            return (i_pnpunct, 1)

    return (i_pnpunct, 0)


def getPosNegEllipsis(input, i_pnpunct):
    output = ''
    count = 0
    input = str(input).lower()
    words = input.split(' ')
    words = re.findall(r"[\w']+|[.:,!?;]", input)

    if '..' not in input:
        return None

    pos_score = 0
    neg_score = 0

    for word in words:
        if word.lower() in sentiment_dict:
            sentiment = sentiment_dict[word.lower()]

            if int(sentiment) == 1:
                #print(word+' found as positive')
                pos_score += 1
            else:
                #print(word+' found as negative')
                neg_score += 1

        if pos_score >= 1 and neg_score == 0:

            return (i_pnpunct, 1)
        elif pos_score == 0 and neg_score >= 1:
            return (i_pnpunct, 1)

    return (i_pnpunct, 0)


def getPunctuation(input, i_excl, i_quest, i_dotdot):
    output = []
    output.append((i_excl, input.count('!')))
    output.append((i_quest, input.count('?')))
    output.append((i_dotdot, input.count('...')))
    return output


def getActions(input):
    output = ''
    words = input.split(' ')
    action = False
    for word in words:
        if '(' in word:
            action = True

        if action:
            output += 'k'+word+' '
        else:
            output += word+' '

        if ')' in word:
            action = False
    return output.strip()


f = open('dataset', 'r', encoding='utf-8-sig')
qid = 0
dict = {}
word_count = {}
rev_dict = {}
index = 1

num_examples = 0

# main method starts
for line in f:
    contents = line.split('\t')
    num_examples += 1
    if len(contents) == 2 and "Scene" not in line:
        dialogue = contents[0].lower()
        #dialogue = dialogue + ' '+ getActions(dialogue).lower()
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


i_quotes = index+1
i_hyp = index+2
i_pnpunct = index+3
i_pnellip = index+4
i_excl = index+5
i_quest = index+6
i_dotdot = index+7
i_interj = index+8


final_features = i_interj
row = np.array((1, final_features))
data = np.zeros((num_examples, final_features))
line_index = 0

print(str(i_excl)+' '+str(i_quest)+' '+str(i_dotdot)+' '+str(i_interj))
f = open('./dataset', 'r', encoding='utf-8-sig')
f_o1 = open('output.txt', 'w')
f_o1.write('# Vocabulary size:'+str(index)+'\n')
for line in f:
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

        first_word = words[0]
        speaker = first_word+':'
        words.append(speaker)

        s_quotes = getQuotes(input, i_quotes)
        s_hyperbole = getHyperbole(input, i_hyp)
        s_pnpunct = getPosNegPunct(input, i_pnpunct)  # handle None case here
        s_pnellip = getPosNegEllipsis(
            input, i_pnellip)  # handle None case here

        s_punct = getPunctuation(line, i_excl, i_quest, i_dotdot)
        s_interj = getInterjection(line, i_interj)

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
        line = []
        s_line = ""
        # print(word_ids)
        for id in word_ids:
            s_line += str(id)+':1 '
            line.append((id, 1))
        line.append(s_quotes)
        if s_hyperbole != None:
            line.append(s_hyperbole)
        if s_pnpunct != None:
            line.append(s_pnpunct)
        line.append(s_interj)
        line.append(s_pnellip)
        line.append(s_punct)
        line.append(s_interj)

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

        line_index += 1
        s_line += str(s_quotes)+' '+str(s_hyperbole) + ' '+str(s_pnpunct) + \
            ' '+str(s_pnellip)+' ' + str(s_punct)+' '+str(s_interj)+' '
        # s_line += '# '+line
        s_line = s_line.strip()
        f_o1.write(str(line)+'\n')
        # print("line break")
print(data[data > 0].size)

data = pd.DataFrame(data)
with open('buschmeier.pkl', 'wb') as f:
    pickle.dump(data, f)

# for test

# f = open(sys.argv[3],'r')
# f_o2 = open(sys.argv[4],'w')

# for line in f:
# 	s_line = ''
# 	contents = line.split('\t')
# 	if "Scene" in line:
# 		qid +=1
# 	pos_score = 0
# 	neg_score = 0

# 	if len(contents) >=2:
# 		#print(contents)
# 		word_ids = [1]
# 		dialogue = contents[0].lower()
# 		#dialogue = dialogue + ' '+ getActions(dialogue).lower()
# 		if len(dialogue) == 0:
# 			continue
# 		words = re.findall(r"[\w']+|[.,!?;]",dialogue)

# 		first_word = words[0]
# 		speaker = first_word+':'
# 		words.append(speaker)

# 		s_quotes = getQuotes(input,i_quotes)
# 		s_hyperbole = getHyperbole(input,i_hyp)
# 		s_pnpunct = getPosNegPunct(input,i_pnpunct)
# 		s_pnellip = getPosNegEllipsis(input,i_pnellip)

# 		s_punct = getPunctuation(line,i_excl,i_quest,i_dotdot)
# 		s_interj = getInterjection(line,i_interj)


# 		for word in words:
# 			if word in dict:
# 				index = dict[word]
# 				if word_count[word] >= 3:
# 					word_ids.append(index)

# 		if contents[1].strip().lower() == 'sarcasm':
# 			label = '+1'
# 		else:
# 			label = '-1'


# 		word_ids = list(set(word_ids))
# 		word_ids.sort()
# 		s_line = label+' '
# 		#print(word_ids)
# 		for id in word_ids:
# 			s_line += str(id)+':1 '

# 		s_line += s_quotes+' '+s_hyperbole +' '+s_pnpunct+' '+s_pnellip+' '
# 		s_line += s_punct+' '+s_interj+' '
# 		s_line += '# '+line
# 		s_line = s_line.strip()
# f_o2.write(s_line+'\n')
