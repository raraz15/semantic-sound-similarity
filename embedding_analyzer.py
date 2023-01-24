import os
import json
import glob
from itertools import combinations

import numpy as np
import pandas as pd

analysis_dir="/home/roguz/freesound-perceptual_similarity/analysis"

if __name__=="__main__":

    embed_paths = glob.glob(analysis_dir+"/*.json")

    # Load the embeddings
    embeddings = {}
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile: # Load the json file
            embedding=json.load(infile)['embeddings']
            if type(embedding) != type(None): # Filter out the None types
                audio_name = os.path.splitext(os.path.basename(embed_path))[0]
                embeddings[audio_name] = np.array(embedding)

    # Create pairs for the remaining
    comb = [(a,b) for a,b in combinations(list(embeddings.keys()), 2)]
    idx = np.arange(len(comb))


    # Compute pairwise dot products
    products = [np.dot(embeddings[a], embeddings[b]) for a,b in comb]
    
    # Sort the products of combinations
    sorted_combs = [(*comb[i],p) for p,i in sorted(zip(products,idx), key=lambda x: x[0], reverse=True)]

    df = pd.DataFrame([{"A": r[0], "B": r[1], "product": r[2]} for r in sorted_combs])
    df.to_csv("analysis_unnorm.csv",index=False)

    products = []
    # Compute pairwise dot products of normalized embeddings
    for a,b in comb:
        emb_a = embeddings[a]/np.linalg.norm(embeddings[a])
        emb_b = embeddings[b]/np.linalg.norm(embeddings[b])
        products.append(np.dot(emb_a,emb_b))

    # Sort the products of combinations
    sorted_combs = [(*comb[i],p) for p,i in sorted(zip(products,idx), key=lambda x: x[0], reverse=True)]

    df = pd.DataFrame([{"A": r[0], "B": r[1], "product": r[2]} for r in sorted_combs])
    df.to_csv("analysis_norm.csv",index=False)