# Sarcasm-Detection
This repo contains all the experiments conducted for sarcasm detection

## Getting Started
1. Lieb Feature (L)
2. Gonzalez Features (G)
3. Bush Features (B)
4. Joshi Congruity Features (J)
5. Variable Window for Word-Embedding Similarity
6. Character Embedding generator
(Need to add citations)

### Prerequisites
sklearn, scipy, gensim

### Installation
Install the packages mentioned above and python 3.6 has been used for all the experiments.

## Feature Generation
Note:- All data is stores as scipy sparse matrices (To enable faster processing and lesser RAM load)

Each subfolder contains a script for each feature generation.
The feature generation steps are explained in respective papers and also we have looked at AdityaJoshi
github repository for reference in terms of implementation. https://github.com/adityajo/ComputationalSarcasm

All the paper named folders generate the respective features.

./wembedding
This folder contains the script for generating word-embedding based similarity features.
You can load the .bin or .txt pre trained vectors like Glove or Word2vec using **gensim**.


Add the train dataset and test dataset path and the required output path.

The dataset SHOULD be in the following format and in .tsv -- MANDATORY for the script to work.
text1<tab>label1
text2<tab>label2
...

After features are generated successfully you can load them and start training.

## Training
Note:- Do mention all paths correctly, there are many and it can get confusing, so please name files appropriately
when storing outputs.

There are a total of 6 scripts each for a set of experiments.
1. final_1 : Training script for experiment including only L features. 
2. final_2 : Training script for experiment including only G features.
3. final_3 : Training script for experiment including only B features.
4. final_4 : Training script for experiment including only J features.
5. b_j_features: Training script for experiment including only B+J features.
6. all_features: Training script for experiment including only L+G+B+J features.

These scripts optionally can include word embedding similarity based features. Just uncomment or load the 
features generated and add it when the features are added using scipy.spare hstack function.

The output stats include Precision, Recall and F-Score values in the specified path.
