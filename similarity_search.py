import os
import time
import argparse
import json
import glob

import numpy as np

ANALYSIS_DIR = "similarity_results"

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

# TODO: Nearest Neighbor, ANN
def find_similar_sounds(query, corpus, N):
    # Compute pairwise dot similarities of normalized embeddings
    similarities = [np.dot(query, ref) for ref in corpus]
    indices = np.argsort(similarities)[::-1][1:N+1] # Top N sounds, except itself
    return similarities, indices

# TODO: for large sound collections, write the output when a row is complete
if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Embedding analyzer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to directory containing embedding json files.')
    parser.add_argument("-a", "-aggregation", type=str, default="mean", help="Type of embedding aggregation.")
    parser.add_argument('-N', type=int, default=25, help="Number of queries to return.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "**", "*.json"), recursive=True)
    print(f"{len(embed_paths)} embeddings were found in the directory.")

    # Load the embeddings and process them
    print("Reading the embeddings and processing them...")
    start_time = time.time()
    embeddings, audio_paths, max_str_len = [], [], 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile: # Load the json file
            model_outputs = json.load(infile)
        if model_outputs['embeddings'] is not None: # Filter out the None types
            # Aggregate frame level embeddings into a clip level embedding
            clip_embedding = aggregate_frames(model_outputs["embeddings"], 
                                              aggregation=args.a)
            embeddings.append(clip_embedding)
            audio_paths.append(model_outputs["audio_path"])
            if len(model_outputs["audio_path"]) > max_str_len: # For pretty print
                max_str_len = len(model_outputs["audio_path"])
    print(f"{len(embeddings)} embeddings were read.")
    total_time = time.time()-start_time
    print(f"Total processing time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    print("\nFor each item, finding similar sounds...")
    start_time = time.time()
    similarity_scores, similarity_indices = [], []
    for i,query in enumerate(embeddings):
        if i%1000==0:
            print(f"[{i+1}/{len(embeddings)}]")
        similarities, indices = find_similar_sounds(query, embeddings, args.N)
        similarity_scores.append(similarities)
        similarity_indices.append(indices)
    total_time = time.time()-start_time
    print(f"Total computation time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(embeddings):.3f} sec.")
    print()

    # Print top args.N sounds for each sound
    string = ""
    for i,(similarities,indices) in enumerate(zip(similarity_scores,similarity_indices)):
        string += f"{'T':>{len(str(args.N))+1}} | {audio_paths[i]}"
        for n,(s,j) in enumerate(zip(similarities,indices)):
            string += f"\n{'Q'+str(n):>{len(str(args.N))+1}} | {audio_paths[j]:<{max_str_len}} | {np.round(s,3)}"
        string += "\n\n"

    # Create the export directory
    embeddings_name = os.path.basename(args.path)
    model_name = os.path.basename(os.path.dirname(args.path))
    export_dir = os.path.join(ANALYSIS_DIR, model_name, embeddings_name)
    print(f"Analysis results are exported to: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    # Export the results
    with open(os.path.join(export_dir, f"{args.a}-results.txt"), "w") as outfile:
        outfile.write(string)

    ##############
    print("Done!")