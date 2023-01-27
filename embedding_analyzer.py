import os
import argparse
import json
import glob
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd

ANALYSIS_DIR = "analysis"

#def path_2_levels(path):
#    one = os.path.dirname(path)
#    two = os.path.dirname(one)
#    return os.path.join(two,one,os.path.basename(path))

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Embedding analyzer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to directory containing embedding json files.')
    parser.add_argument('-N', type=int, default=3, help="Number of queries to return.")
    #parser.add_argument('-o', '--output-dir', type=str, default=ANALYSIS_DIR, help="Save output files to a directory. If none specified, saved next to inputs.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "**", "*.json"))

    # Create the export directory
    embeddings_name = os.path.basename(args.path)
    export_dir = os.path.join(ANALYSIS_DIR, embeddings_name)
    os.makedirs(export_dir, exist_ok=True)

    # Load the embeddings
    embeddings, max_str_len = {}, 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile: # Load the json file
            embedding=json.load(infile)['embeddings']
            if type(embedding) != type(None): # Filter out the None types
                #audio_name = os.path.splitext(os.path.basename(embed_path))[0]
                embeddings[embed_path] = np.array(embedding)
                if len(embed_path) > max_str_len: # For pretty print
                    max_str_len = len(embed_path)
    embed_paths = list(embeddings.keys()) # Overwrite
    embeds = list(embeddings.values())

    # Compute pairwise dot products of normalized embeddings
    products = np.zeros((len(embeds),len(embeds))) # Encode 0 for similarity to itself
    for i,embed_a in enumerate(embeds):
        for j,embed_b in enumerate(embeds):
            if i<=j:
                continue
            embed_a = embed_a/np.linalg.norm(embed_a)
            embed_b = embed_b/np.linalg.norm(embed_b)
            products[i,j] = np.dot(embed_a,embed_b)
            products[j,i] = products[i,j]

    # Print top args.N sounds for each sound
    for i,row in enumerate(products):
        query_sound_path = os.path.splitext(embed_paths[i])[0]
        indices = np.argsort(row)[::-1][:args.N] # Top 3 sounds
        print(f"\n{query_sound_path}".replace("embeddings","sounds"))
        for j in indices:
            match_sound_path = os.path.splitext(embed_paths[j])[0]
            print(f"{match_sound_path:<{max_str_len-4}} - {np.round(row[j],3)}".replace("embeddings","sounds"))




    #products = []
    ## Compute pairwise dot products of normalized embeddings
    #for a, b in comb:
    #    emb_a = embeddings[a]/np.linalg.norm(embeddings[a])
    #    emb_b = embeddings[b]/np.linalg.norm(embeddings[b])
    #    products.append(np.dot(emb_a, emb_b))

    ## Sort the products of combinations
    #sorted_combs = [(*comb[i],p) for p,i in sorted(zip(products,idx), key=lambda x: x[0], reverse=True)]
#
    #df = pd.DataFrame([{"A": r[0], "B": r[1], "product": r[2]} for r in sorted_combs])
    #df.to_csv(os.path.join(export_dir, "analysis_norm.csv"),index=False)

    ## Compute pairwise dot products
    #products = [np.dot(embeddings[a], embeddings[b]) for a,b in comb]
    #
    ## Sort the products of combinations
    #sorted_combs = [(*comb[i],p) for p,i in sorted(zip(products,idx), key=lambda x: x[0], reverse=True)]
#
    #df = pd.DataFrame([{"A": r[0], "B": r[1], "product": r[2]} for r in sorted_combs])
    #df.to_csv(os.path.join(export_dir, "analysis_unnorm.csv"),index=False)