import pandas as pd
import numpy as np
import gensim
import pickle
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def wembed_util(matrix):
    sim_matrix = cosine_similarity(matrix)
    # Replace Extremely Similar Words with 0
    sim_matrix[sim_matrix == 0] = 0.001
    sim_matrix[sim_matrix > 0.99] = 0
    N = sim_matrix.shape[0]
    
    # Similarity Features
    sim_df = pd.DataFrame(sim_matrix)
    sim_df = sim_df.replace(0, np.nan)
    sim = sim_df.max(axis=1)
    dsim = sim_df.min(axis=1)
    
    # Weighted Similarity Features
    pos = [[i] for i in range(0, N)]
    weights = euclidean_distances(pos, squared=True)
    weights = pd.DataFrame(weights)
    weights = weights.replace(0, 1)
    
    wsim_df = pd.DataFrame(sim_matrix/weights).replace(np.Inf, np.nan)
    wsim_df = wsim_df.replace(0, np.nan)
    wsim  = wsim_df.max(axis=1)
    wdsim  = wsim_df.min(axis=1)
    
    
    # Extract 8 features
    ff = [sim.max(axis=0), sim.min(axis=0), dsim.max(axis=0), dsim.min(axis=0), 
          wsim.max(axis=0), wsim.min(axis=0), wdsim.max(axis=0), wdsim.min(axis=0)]
    
    return ff

def wembed_features(df, model, tokenizer, window = 3):
    # Features Matrix
    features = []
    
    for ind, row in df.iterrows():
        # Get text 
        string = row['text']
        
        # Tokenize
        tokens = tokenizer.tokenize(string)
        tokens = list(tokens)
        tokens = [x.lower() for x in tokens]
#         tokens = [lemmatizer.lemmzatize(tok) for tok in tokens]
        
        token_vectors = []
        accepted_tokens = []
        for i in range(int(window/2)):
            token_vectors.append(np.zeros(model.vector_size))
            accepted_tokens.append('null')
        for token in tokens:
            try:
                token_vectors.append(model.word_vec(token))
                accepted_tokens.append(token)
            except Exception as e:
                token_vectors.append(np.random.rand(model.vector_size))
                accepted_tokens.append(token+'#rand')
         
        for i in range(int(window/2)):
            token_vectors.append(np.zeros(model.vector_size))
            accepted_tokens.append('null')
        
        # Window Buffer 
        last = 0
        vector_size = model.vector_size
        window_buffer = np.zeros((window, vector_size))
        
        # Final Vector List
        final_vectors = []
        final_tokens = []
        for vector in token_vectors:
            if last < window:
                # Update Buffer with new vector
                window_buffer[last, :] = vector
                # If window is full
                if last == window-1:
                    new_vec = window_buffer.mean(axis=0)
                    final_vectors.append(new_vec)
                    final_tokens.append('-'.join(accepted_tokens[0:3]))
                    
                last += 1
            
            else:
                if window == 1:
                    next_pos = 0
                else:
                    next_pos = (last%window)
                
                window_buffer[next_pos, :] = vector
                
                new_vec = window_buffer.mean(axis=0)
                final_vectors.append(new_vec)
                final_tokens.append('-'.join(accepted_tokens[last+1-window:last+1]))
                
                last += 1
                
                # End of Buffer
                if last == len(token_vectors):
                    break
        
        # Free Up Memory
        del window_buffer
        del accepted_tokens
        del token_vectors
        
        # If final_vectors is empty fill zeros
        if len(final_tokens) < 2:
            features.append([0 for i in range(0, 8)])
        else: 
            features.append(wembed_util(final_vectors))
        
        print(ind)
        
    features_df = pd.DataFrame(features, columns=['max_sim', 'min_sim', 'max_dsim', 'min_dsim', 'max_wsim', 'min_wsim', 'max_wdsim', 'min_wdsim' ])
    if len(features_df.isnull().any(1).nonzero()[0]) == 0:
        print("No Nans")
    else:
        features_df = features_df.fillna(0) 
    
    return features_df    

print('Finished Importing')

# Load Pretrained Vectors
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Loaded Model')

# Tokenizer
tokenizer = RegexpTokenizer(r"[\w']+|[.:,!?;]")

# Load Dataset
df = pd.read_csv('../final_data.tsv', sep='\t')

feat_df = wembed_features(df, word2vec_model, tokenizer, window=3)
print('Extracted Features')
feat_df.to_pickle('wembed_3.pkl')