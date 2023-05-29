"""In a corpus of sound embeddings, takes each sound as a query and 
searches for similar sounds using user defined strategies."""

import os
import time
import json
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from directories import ANALYSIS_DIR

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
# TODO: output format should be a list of lists in a dict
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', 
                        type=str, 
                        required=True, 
                        help='Directory containing embedding.json files.')
    parser.add_argument("-s", "--search", 
                        type=str, 
                        choices=["dot", "nn"], 
                        default="dot", 
                        help="Type of similarity search algorithm.")
    parser.add_argument('-N', 
                        type=int, 
                        default=90, 
                        help="Number of queries to return.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "*.json"))
    print(f"{len(embed_paths)} embeddings were found in the directory.")

    # Load the embeddings
    embeddings, audio_paths, str_len = [], [], 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile:
            clip_embedding = json.load(infile)
        embeddings.append(np.array(clip_embedding["embeddings"]))
        audio_paths.append(clip_embedding["audio_path"])
        # For pretty print
        if len(clip_embedding["audio_path"]) > str_len:
            str_len = len(clip_embedding["audio_path"])
    print(f"{len(embeddings)} embeddings were read.")

    # Perform the sound search
    print("For each sound, searching for similar sounds...")
    start_time = time.time()
    similarity_scores, similarity_indices = [], []
    for i,query in enumerate(embeddings):
        if i%1000==0:
            print(f"[{i:>{len(str(len(embeddings)))}}/{len(embeddings)}]")
        similarities, indices = search_similar_sounds(query, embeddings, args.N, args.search)
        similarity_scores.append(similarities)
        similarity_indices.append(indices)
    total_time = time.time()-start_time
    print(f"Total computation time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(embeddings):.3f} sec.")

    # Create the export directory
    model_name = os.path.basename(args.path)
    dataset_name = os.path.basename(os.path.dirname(args.path))
    output_dir = os.path.join(ANALYSIS_DIR, dataset_name, model_name, args.search)
    output_path = os.path.join(output_dir, "similarity_results.json")
    print(f"Exporting analysis results to: {output_path}")
    os.makedirs(output_dir, exist_ok=True)

    # Export results to a json file
    results_dict = {}
    for i,(similarities,indices) in enumerate(zip(similarity_scores,similarity_indices)):
        query_fname = os.path.splitext(os.path.basename(audio_paths[i]))[0]
        results_dict[query_fname] = []
        for j in indices:
            score = similarities[j]
            ref_fname = os.path.splitext(os.path.basename(audio_paths[j]))[0]
            results_dict[query_fname].append({"file_name": ref_fname, 
                                              "score": float(score)})
    with open(output_path, "w") as outfile:
        json.dump(results_dict, outfile, indent=4)

    ##############
    print("Done!")