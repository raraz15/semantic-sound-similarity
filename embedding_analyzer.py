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
    print(f"{len(embed_paths)} embeddings were found.")

    # Load the embeddings
    embeddings, max_str_len = [], 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile: # Load the json file
            model_outputs = json.load(infile)
        if model_outputs['embeddings'] is not None: # Filter out the None types
            del model_outputs["classes"] # Remove unnecessary info
            del model_outputs["top_10_classes_probabilities"]
            embeddings.append(model_outputs)
            if len(model_outputs["audio_path"]) > max_str_len: # For pretty print
                max_str_len = len(model_outputs["audio_path"])
    print(f"{len(embeddings)} embeddings were read.")

    # Compute pairwise dot products of normalized embeddings
    products = np.zeros((len(embeddings),len(embeddings))) # Encode 0 for similarity to itself
    for i,embed_a in enumerate(embeddings):
        embed_a = embed_a["embeddings"]/np.linalg.norm(embed_a["embeddings"])
        for j,embed_b in enumerate(embeddings):
            if i<=j:
                continue
            embed_b = embed_b["embeddings"]/np.linalg.norm(embed_b["embeddings"])
            products[i,j] = np.round(np.dot(embed_a,embed_b),4) # Round the dot product
            products[j,i] = products[i,j]

    # Create the export directory
    embeddings_name = os.path.basename(args.path)
    export_dir = os.path.join(ANALYSIS_DIR, embeddings_name)
    print(f"Analysis results will be exported to: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    # Print top args.N sounds for each sound
    string = ""
    for i,row in enumerate(products):
        string += f"Target: {embeddings[i]['audio_path']}"
        indices = np.argsort(row)[::-1][:args.N] # Top 3 sounds
        for n,j in enumerate(indices):
            string += f"\nQ{n} | {embeddings[j]['audio_path']:<{max_str_len-4}} | {np.round(row[j],3)}"
        string += "\n\n"

    # Export the results
    with open(os.path.join(export_dir, "results.txt"), "w") as outfile:
        outfile.write(string)

    ##############
    print("Done!")