import os
import argparse
import json
import glob

import numpy as np

ANALYSIS_DIR = "analysis"

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Embedding analyzer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to directory containing embedding json files.')
    parser.add_argument('-N', type=int, default=3, help="Number of queries to return.")
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
    products=np.round(products,3) # round to 3 decimal points

    # Print top args.N sounds for each sound
    string = ""
    for i,row in enumerate(products):
        query_sound_path = os.path.splitext(embed_paths[i])[0]
        indices = np.argsort(row)[::-1][:args.N] # Top 3 sounds
        string += f"Target: {query_sound_path}".replace("embeddings","sounds")
        for n,j in enumerate(indices):
            match_sound_path = os.path.splitext(embed_paths[j])[0]
            string += f"\nQ{n} | {match_sound_path:<{max_str_len-4}} | {np.round(row[j],3)}".replace("embeddings","sounds")
        string += "\n\n"
    
    # Export the results
    with open("results.txt", "w") as outfile:
        outfile.write(string)