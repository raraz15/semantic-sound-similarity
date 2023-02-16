import os
import time
import argparse
import json
import glob

import numpy as np

ANALYSIS_DIR = "analysis"

# TODO: frame aggregation
# TODO: energy based frame filtering
# TODO: PCA
def aggregate_frames(embeds, normalize=True, aggregation="none"):
    """ Takes a list of embeddings and aggregates them into a clip level embedding.
    """
    # Convert to numpy array
    if type(embeds)==list:
        embeds = np.array(embeds)
    # Normalize each time frame by itself if specified
    if normalize:
        embeds = embeds/np.linalg.norm(embeds,axis=1)[..., np.newaxis]
    # Aggreagate
    if aggregation=="mean":
        embeds = embeds.mean(axis=0)
    elif aggregation=="median":
        embeds = np.median(embeds, axis=0)
    return embeds

# TODO: for large sound collections, write the output when a row is complete
if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Embedding analyzer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to directory containing embedding json files.')
    parser.add_argument("-a", "-aggregation", type=str, default="mean", help="Type of embedding aggregation.")
    parser.add_argument('-N', type=int, default=10, help="Number of queries to return.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "**", "*.json"), recursive=True)
    print(f"{len(embed_paths)} embeddings were found.")

    # Load the embeddings
    embeddings, max_str_len = [], 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile: # Load the json file
            model_outputs = json.load(infile)
        if model_outputs['embeddings'] is not None: # Filter out the None types
            # Aggregate frame level embeddings into a clip level embedding
            clip_embedding = aggregate_frames(model_outputs["embeddings"], 
                                              aggregation=args.a)
            embeddings.append({"audio_path": model_outputs["audio_path"], 
                                "embeddings": clip_embedding})
            if len(model_outputs["audio_path"]) > max_str_len: # For pretty print
                max_str_len = len(model_outputs["audio_path"])
    print(f"{len(embeddings)} embeddings were read.")

    # TODO: Nearest Neighbor, ANN
    # Compute pairwise dot products of normalized embeddings
    print("Computing pairwise dot products...")
    start_time = time.time()
    products = np.zeros((len(embeddings),len(embeddings))) # Encode 0 for similarity to itself
    for i,embed_a in enumerate(embeddings):
        embed_a = embed_a["embeddings"]/np.linalg.norm(embed_a["embeddings"])
        for j,embed_b in enumerate(embeddings):
            if i<=j:
                continue
            embed_b = embed_b["embeddings"]/np.linalg.norm(embed_b["embeddings"])
            products[i,j] = np.round(np.dot(embed_a,embed_b),4) # Round the dot product
            products[j,i] = products[i,j]
    total_time = time.time()-start_time
    print(f"\nTotal computation time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Create the export directory
    embeddings_name = os.path.basename(args.path)
    model_name = os.path.basename(os.path.dirname(args.path))
    export_dir = os.path.join(ANALYSIS_DIR, model_name, embeddings_name)
    print(f"Analysis results will be exported to: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    # Print top args.N sounds for each sound
    string = ""
    for i,row in enumerate(products):
        string += f"T  | {embeddings[i]['audio_path']}"
        indices = np.argsort(row)[::-1][:args.N] # Top 3 sounds
        for n,j in enumerate(indices):
            string += f"\nQ{n} | {embeddings[j]['audio_path']:<{max_str_len}} | {np.round(row[j],3)}"
        string += "\n\n"

    # Export the results
    with open(os.path.join(export_dir, "results.txt"), "w") as outfile:
        outfile.write(string)

    ##############
    print("Done!")