from collections import defaultdict
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import random, pickle

def generate_se_embedding(vec_path):
    """Generates side-effect embeddings. 

    Parameters:
    vec_path (str): Path to the BioWordVec vector file (BioWordVec_PubMed_MIMICIII_d200.vec.bin), downloaded from https://github.com/ncbi-nlp/BioSentVec.
    """
    embed = Embedding(path=vec_path)
    with open('datasets/list_sideeffects.txt') as f:
        sideEffects = f.read().split('\n')
        sideEffects.pop()
        
        df = { 'sideEffectID' : [], 'embedding' : [] }  
        
        for i, sideEffect in enumerate(sideEffects):
            formatted = sideEffect.replace('.', ' ')
            if formatted == "Stevens   Johnson syndrome": formatted = "Stevens - Johnson syndrome" # This might be a typo in ID.txt. Accounting for it here instead of changing the .txt file
            embedding = embed.get_embedding(formatted)
            df['sideEffectID'].append(i); df['embedding'].append(embedding)
            
        df = pd.DataFrame(df)
        df.to_csv('datasets/embeds/side_effect_embed.csv')

def generate_soc_data(type="", aggregate="micro", randomise=""):
    """Generates SOC labels and embeddings.

    Parameters:
    type (str): Either "" (default), "weighted", or "full", referencing Appendix B: 'SOC Labelling and Cheated Models' in the paper.
    aggregate (str): Either "micro" or "macro", for micro-averaging or macro-averaging the SOC embeddings respectively.
    randomise (str): "" (no randomisation), "randomised", or "true_randomised", referencing Section 3.2: 'Label Taxonomy Randomisation' in the paper.
    """
    generate_soc_labels(type=type, randomise=randomise)
    generate_soc_embedding(aggregate=aggregate, randomise=f"_{randomise}" if randomise else randomise)

def generate_soc_labels(type="", randomise=""):    
    randomised_path = f"_{randomise}" if randomise else ""
    type_path = f"_{type}" if type else ""
    
    with open('datasets/labels/side_effect_labels.txt') as f, open('datasets/compounds/ID.txt') as d:
        drugIDs = d.read().split('\n')
        drugIDs.pop()
        
        rows = f.read().split('\n')
        rows.pop()
        
        headers = [int(i) for i in rows.pop(0).split('\t')]
        if randomise == "randomised": 
            random.shuffle(headers)
        elif randomise == "true_randomised":
            headers = [random.randint(1, 26) for _ in range(len(headers))]

        with open(f"datasets/labels/soc_headers{randomised_path}.pkl", "wb") as p:
            pickle.dump(headers, p)

    df = defaultdict(list)

    for i, row in enumerate(rows):
        df['drugID'].append(drugIDs[i])
        for j in range(1,27):
            df[j].append(0)
            
        row = row.split('\t')
        if type == "weighted":
            curr_counts = { i:0 for i in range(1,27) }
            for j, val in enumerate(row):
                if val == '1':
                    soc = headers[j]
                    curr_counts[int(soc)] += 1

            total_counts = sum(v for v in curr_counts.values())
            for soc, count in curr_counts.items():
                df[soc][-1] = count/total_counts
        elif type =="full":
            for j, val in enumerate(row):
                if val == '1':
                    df[headers[j]][-1] += 1
        else:
            for j, val in enumerate(row):
                if val == '1':
                    df[headers[j]][-1] = 1 

    df = pd.DataFrame(df)
    df.to_csv(f'datasets/labels/soc_labels{type_path}{randomised_path}.csv')

def generate_soc_embedding(aggregate="micro", randomise=""):
    with open('datasets/labels/side_effect_labels.txt') as l, open('datasets/embeds/side_effect_embed.csv') as e, open(f'datasets/labels/soc_headers{randomise}.pkl', 'rb') as h:
        se_embeddings = pd.read_csv(e)
        headers = pickle.load(h)
        rows = l.read().split('\n')
        rows.pop(); rows.pop(0)

    map = { i:np.zeros(200) for i in range(26) } 
    counts = [0 for i in range(26)]
    if aggregate == "micro":
        for i, row in enumerate(rows):
            row = row.split('\t')
            for j, val in enumerate(row):
                if val == '1':
                    curr_embedding = se_embeddings.iloc[j]['embedding']
                    curr_embedding = curr_embedding.strip("[]").split()
                    curr_embedding = np.array([float(k) for k in curr_embedding], dtype=np.dtype('float64'))

                    curr_soc = headers[j]-1
                    map[curr_soc] = np.add(map[curr_soc], curr_embedding)
                    counts[curr_soc] += 1

        df = { 'embedding' : [] }
        for i, embedding in map.items():
            df['embedding'].append(np.divide(embedding, counts[i]))

        df = pd.DataFrame(df)
    elif aggregate == "macro":
        for i, val in enumerate(headers):
            curr_embedding = se_embeddings.iloc[i]['embedding']
            curr_embedding = curr_embedding.strip("[]").split()
            curr_embedding = np.array([float(j) for j in curr_embedding], dtype=np.dtype('float64'))

            curr_soc = val-1
            map[curr_soc] = np.add(map[curr_soc], curr_embedding)
            counts[curr_soc] += 1

        df = { 'embedding' : [] }
        for i, embedding in map.items():
            df['embedding'].append(np.divide(embedding, counts[i]))
        
        df = pd.DataFrame(df)

    df.to_csv(f'datasets/embeds/soc_embed_{aggregate}{randomise}.csv')

class Embedding:
    def __init__(self, path):
        # 事前学習済みWord2Vecモデルを読み込む
        self.path = path
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def get_embedding(self, sentence):
        # 文内の単語ベクトルのリストを取得
        word_vectors = [self.model[word] for word in sentence if word in self.model.key_to_index]
        
        # ベクトルが空のリストでない場合、平均ベクトルを計算
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            # 有効な単語がない場合はゼロベクトルを返す
            return np.zeros(self.model.vector_size)

    