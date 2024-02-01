"""In a corpus of sound embeddings, takes each sound as a query and 
searches for similar sounds using user defined strategies."""

import os
import time
import json
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from lib.search_algorithms import search_similar_sounds
from lib.utils import get_fname
from lib.directories import ANALYSIS_DIR

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
                        default="nn", 
                        help="Type of similarity search algorithm.")
    parser.add_argument('-N', 
                        type=int, 
                        default=15, 
                        help="Number of search results per query to save.")
    parser.add_argument("--ground-truth",
                        type=str,
                        default=None,
                        help="Path to the ground truth CSV file. "
                        "You can provide a subset of the ground truth by "
                        "filtering the CSV file before passing it to this script.")
    parser.add_argument("--output-dir",
                        "-o",
                        type=str,
                        default=ANALYSIS_DIR,
                        help="Path to the output directory.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.embed_dir, "*.json"))
    assert len(embed_paths)>0, "No embedding files were found in the directory."
    print(f"{len(embed_paths):,} embedding paths were found in the directory.")

    # Get the ground truth file if provided
    if args.ground_truth is not None:
        print("Reading the ground truth file...")
        # Read the ground truth annotations
        import pandas as pd
        df = pd.read_csv(args.ground_truth)
        fnames = set(df["fname"].to_list())
        # Filter the embeddings to only include the ones in the ground truth
        embed_paths = [embed_path for embed_path in embed_paths if int(get_fname(embed_path)) in fnames]
        assert len(embed_paths)>0, "No embedding files are referenced in the ground truth file."
        print(f"{len(embed_paths)} embeddings are in the ground truth file.")

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
    print(f"{len(embeddings):,} embeddings are read successfully.")

    # Create the export directory
    args.embed_dir = os.path.normpath(args.embed_dir)
    model_name = os.path.basename(args.embed_dir)
    dataset_name = os.path.basename(os.path.dirname(args.embed_dir))
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, args.search)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "similarity_results.json")
    print(f"Analysis results will be exported to: {output_path}")

    # For each element in the dataset, perform the sound search to the rest of the dataset
    print("Making similarity search queries for each embedding...")
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
            if (i+1)%1000==0 or (i+1)==len(embeddings) or i==0:
                print(f"[{i+1:>{len(str(len(embeddings)))}}/{len(embeddings)}]")
    total_time = time.monotonic()-start_time
    print(f"Total computation time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/query: {total_time/len(embeddings):.3f} sec.")

    ##############
    print("Done!")