"""In a corpus of sound embeddings, takes each sound as a query and 
searches for similar sounds using user defined strategies."""

import os
import time
import argparse
import json
import glob

import numpy as np

ANALYSIS_DIR = "data/similarity_results"

# In a preproccesing script,
# TODO: energy based frame filtering (at audio input)
# TODO: PCA

def aggregate_frames(embeds, normalize=True, aggregation="mean"):
    """ Takes a list of embeddings and aggregates them into a 
    clip level embedding if not already aggregated.
    """
    # Convert to numpy array
    if type(embeds)==list:
        embeds = np.array(embeds)
    # Aggreagate if multiple frames exist and specified
    if aggregation!="none" and len(embeds.shape)!=1:
        if aggregation=="mean":
            embeds = embeds.mean(axis=0)
        elif aggregation=="median":
            embeds = np.median(embeds, axis=0)
        elif aggregation=="max":
            embeds = embeds.max(axis=0)
    # Normalize the clip level embedding
    if normalize:
        embeds = embeds/np.linalg.norm(embeds)
    return embeds

def dot_product_search(query, corpus, N):
    """Computes pairwise dot product similarities and returns the indices of top N"""
    assert len(query.shape)==1, f"To use dot product search, queries should be aggregated! {query.shape}"
    similarities = [np.dot(query, ref) for ref in corpus]
    indices = np.argsort(similarities)[::-1][1:N+1] # Do not return itself
    return similarities, indices

# TODO: ANN
def nn_search(query, corpus, N):
    """Computes pairwise distances and returns the indices of bottom N"""
    distances = [np.linalg.norm(query-ref) for ref in corpus]
    indices = np.argsort(distances)[1:N+1] # Do not return itself
    return distances, indices

def search_similar_sounds(query, corpus, N, algo="dot"):
    if algo=="dot":
        return dot_product_search(query, corpus, N)
    elif algo=="nn":
        return nn_search(query, corpus, N)
    else:
        raise NotImplementedError

# TODO: for large sound collections, write the output when a row is complete
# TODO: delete text output, only json
if __name__=="__main__":

    parser=argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Directory containing embedding json files.')
    parser.add_argument("-a", "-aggregation", 
                        type=str, 
                        choices=["mean", "median", "max", "none"], 
                        default="mean", 
                        help="Type of embedding aggregation.")
    parser.add_argument("-s", "--search", 
                        type=str, 
                        choices=["dot", "nn"],
                        default="dot", 
                        help="Type of similarity search algorithm.")
    parser.add_argument('-N', 
                        type=int, 
                        default=205, 
                        help="Number of queries to return.")
    parser.add_argument("--no-normalization",
                        action="store_false", 
                        help="Do not normalize the aggregated embeddings.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "**", "*.json"), recursive=True)
    print(f"{len(embed_paths)} embeddings were found in the directory.")

    # Load the embeddings and process them
    print("Reading the embeddings and pre-processing them...")
    start_time = time.time()
    embeddings, audio_paths, str_len = [], [], 0
    for embed_path in embed_paths:
        # Load the json file
        with open(embed_path, 'r') as infile:
            model_outputs = json.load(infile)
        # Process and collect
        if model_outputs['embeddings'] is not None: # Filter out the None types
            clip_embedding = aggregate_frames(model_outputs["embeddings"], 
                                              normalize=args.no_normalization, 
                                              aggregation=args.a)
            embeddings.append(clip_embedding)
            audio_paths.append(model_outputs["audio_path"])
            # For pretty printing
            if len(model_outputs["audio_path"]) > str_len:
                str_len = len(model_outputs["audio_path"])
    print(f"{len(embeddings)} embeddings were read.")
    total_time = time.time()-start_time
    print(f"Total pre-processing time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Perform the sound search
    print("\nFor each sound, searching for similar sounds...")
    start_time = time.time()
    similarity_scores, similarity_indices = [], []
    for i,query in enumerate(embeddings):
        if i%1000==0:
            print(f"[{i:>{len(str(1000))}}/{len(embeddings)}]")
        similarities, indices = search_similar_sounds(query, embeddings, args.N, args.search)
        similarity_scores.append(similarities)
        similarity_indices.append(indices)
    total_time = time.time()-start_time
    print(f"Total computation time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(embeddings):.3f} sec.")

    # Create the export directory
    embeddings_name = os.path.basename(args.path)
    model_name = os.path.basename(os.path.dirname(args.path))
    export_dir = os.path.join(ANALYSIS_DIR, model_name, embeddings_name)
    output_path = os.path.join(export_dir, f"{args.search}-{args.a}-results.json")
    print(f"\nExporting analysis results to: {output_path}")
    os.makedirs(export_dir, exist_ok=True)

    # Export results to a json file
    results_dict = {}
    for i,(similarities,indices) in enumerate(zip(similarity_scores,similarity_indices)):
        query_fname = os.path.splitext(os.path.basename(audio_paths[i]))[0]
        results_dict[query_fname] = []
        for n,j in enumerate(indices):
            score = similarities[j]
            ref_fname = os.path.splitext(os.path.basename(audio_paths[j]))[0]
            results_dict[query_fname].append({ref_fname: float(score)})
    with open(output_path, "w") as outfile:
        json.dump(results_dict, outfile, indent=4)

    ## Write the top args.N sounds for each sound to a text file
    #string = ""
    #indent = len(str(args.N))+1 # pretty print
    #for i,(similarities,indices) in enumerate(zip(similarity_scores,similarity_indices)):
    #    string += f"{'T':>{indent}} | {audio_paths[i]}"
    #    for n,j in enumerate(indices):
    #        s = np.round(similarities[j],3) # round for display
    #        string += f"\n{'Q'+str(n):>{indent}} | {audio_paths[j]:<{str_len}} | {s}"
    #    string += "\n\n"
    #with open(os.path.join(export_dir, f"{args.search}-{args.a}-results.txt"), "w") as outfile:
    #    outfile.write(string)

    ##############
    print("Done!")