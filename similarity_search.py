"""In a corpus of sound embeddings, takes each sound as a query and 
searches for similar sounds using user defined strategies."""

import os
import time
import json
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd

from directories import ANALYSIS_DIR, GT_PATH, EVAL_DIR

def get_fname(audio_path):
    """Returns the file name without the extension."""
    return os.path.splitext(os.path.basename(audio_path))[0]

def dot_product_search(query, corpus, N):
    """Computes pairwise dot product similarities and returns the indices of top N. 
    Assumes that the query is aggregated and the query is removed from the corpus. 
    Returns a dictionary with the query fname and a list of dictionaries with 
    the results and their scores."""

    query_embed, query_path = query[0], query[1]
    assert len(query_embed.shape)==1, f"To use dot product search, queries should be aggregated! {query_embed.shape}"
    # For each reference in the dataset, compute the dot product with the query
    products = [np.dot(query_embed, ref[0]) for ref in corpus]
    # Get the indices of the top N similar elements in the corpus
    indices = np.argsort(products)[::-1][:N]
    # Return the results
    return {"query_fname": get_fname(query_path), 
            "results": [{"result_fname": get_fname(corpus[i][1]), 
                         "score": products[i]} for i in indices],
            "search": "dot_product"
            }

# TODO: ANN
def nn_search(query, corpus, N):
    """Computes pairwise distances and returns the indices of bottom N. 
    Assumes that the query is aggregated and the query is removed from the corpus. 
    Returns a dictionary with the query fname and a list of dictionaries with 
    the results and their scores."""

    query_embed, query_path = query[0], query[1]
    # For each reference in the dataset, compute the distance to the query
    distances = [np.linalg.norm(query_embed-ref[0]) for ref in corpus]
    # Get the indices of the top N closest elements in the corpus
    indices = np.argsort(distances)[:N]
    # Return the results
    return {"query_fname": get_fname(query_path), 
            "results": [{"result_fname": get_fname(corpus[i][1]), 
                         "score": distances[i]} for i in indices],
            "search": "nearest_neighbour"
            }

def search_similar_sounds(query, corpus, N, algo="dot"):
    if algo=="dot":
        return dot_product_search(query, corpus, N)
    elif algo=="nn":
        return nn_search(query, corpus, N)
    else:
        raise NotImplementedError

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('embed_dir', 
                        type=str, 
                        help='Directory containing embedding.json files. ' 
                        'Embeddings should be prepared with create_clip_level_embedding.py.')
    parser.add_argument("-s", 
                        "--search", 
                        type=str, 
                        choices=["dot", "nn"], 
                        default="dot", 
                        help="Type of similarity search algorithm.")
    parser.add_argument('-N', 
                        type=int, 
                        default=30, 
                        help="Number of queries to return.")
    parser.add_argument("--ground-truth",
                        type=str,
                        default=GT_PATH,
                        help="Path to the ground truth CSV file. "
                        "You can provide a subset of the ground truth by "
                        "filtering the CSV file before passing it to this script.")
    parser.add_argument("--output-dir",
                        type=str,
                        default=ANALYSIS_DIR,
                        help="Path to the output directory.")
    args=parser.parse_args()

    # Read the ground truth annotations
    df = pd.read_csv(args.ground_truth)
    fnames = set(df["fname"].to_list())

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.embed_dir, "*.json"))
    print(f"{len(embed_paths)} embedding paths were found in the directory.")
    # Filter the embeddings to only include the ones in the ground truth
    embed_paths = [embed_path for embed_path in embed_paths if int(get_fname(embed_path)) in fnames]
    print(f"{len(embed_paths)} embeddings are in the ground truth.")

    # Load the embeddings, convert to numpy and store with the audio path
    print("Loading the embeddings...")
    embeddings, str_len = [], 0
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile:
            clip_embedding = json.load(infile)
        embeddings.append((np.array(clip_embedding["embeddings"]), clip_embedding["audio_path"]))
        # For pretty print
        if len(clip_embedding["audio_path"]) > str_len:
            str_len = len(clip_embedding["audio_path"])
    print(f"{len(embeddings)} embeddings are read.")

    # Create the export directory
    model_name = os.path.basename(args.embed_dir)
    dataset_name = os.path.basename(os.path.dirname(args.embed_dir))
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, args.search)
    output_path = os.path.join(output_dir, "similarity_results.json")
    print(f"Analysis results will be exported to: {output_path}")
    os.makedirs(output_dir, exist_ok=True)

    # For each element in the dataset, perform the sound search to the rest of the dataset
    print("For each sound in the dataset, searching for similar sounds...")
    start_time = time.monotonic()
    with open(output_path, "w") as outfile:
        for i in range(len(embeddings)):
            # Remove the query from the corpus
            _embeddings = embeddings.copy()
            query = _embeddings.pop(i)
            # Compare the query to the rest of the corpus
            results = search_similar_sounds(query, _embeddings, args.N, args.search)
            # Export the results to JSONL file
            outfile.write(json.dumps(results)+"\n")
            # Display progress
            if (i+1)%1000==0 or (i+1)==len(embeddings):
                print(f"[{i+1:>{len(str(len(embeddings)))}}/{len(embeddings)}]")
    total_time = time.monotonic()-start_time
    print(f"Total computation time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(embeddings):.3f} sec.")

    ##############
    print("Done!")