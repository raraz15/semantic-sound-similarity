""" Since calculation of R1 requires the complete ranking for each query,
instead of first running similarity_search.py with N=-1 and storing the results,
here we first do a similarity_search with N=-1 without saving the results
 and then calculate the R1 for each query."""

import os
import time
import json
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd

from similarity_search import get_fname
from lib.directories import ANALYSIS_DIR, GT_PATH, TAXONOMY_FAMILY_JSON
import lib.metrics as metrics

METRICS = ["micro_mr1", "macro_mr1"]

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
    parser.add_argument('--metrics',
                        type=str,
                        nargs='+',
                        default=METRICS, 
                        help='Metrics to calculate.')
    parser.add_argument("--ground-truth",
                        type=str,
                        default=GT_PATH,
                        help="Path to the ground truth CSV file. "
                        "You can provide a subset of the ground truth by "
                        "filtering the CSV file before passing it to this script.")
    parser.add_argument("--families-json",
                        type=str,
                        default=TAXONOMY_FAMILY_JSON,
                        help="Path to the JSON file containing the family information "
                        "of the FSD50K Taxonomy. You can also provide the family information from "
                        "an ontology")
    parser.add_argument("--output-dir",
                        type=str,
                        default=ANALYSIS_DIR,
                        help="Path to the output directory.")
    args=parser.parse_args()

    # Check the metrics
    args.metrics = [metric.lower() for metric in args.metrics]
    assert set(args.metrics).issubset(set(METRICS)), \
        f"Invalid metrics. Valid metrics are: {METRICS}"

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
    embeddings = {}
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile:
            clip_embedding = json.load(infile)
        fname = get_fname(embed_path)
        embeddings[fname] = np.array(clip_embedding["embeddings"])
    print(f"{len(embeddings)} embeddings are read.")

    # Determine the output directory
    model_name = os.path.basename(args.embed_dir)
    dataset_name = os.path.basename(os.path.dirname(args.embed_dir))
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, args.search)
    print(f"Results will be exported to {output_dir}")
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Micro Averaged R1 (MR1) if required
    if "micro_mr1" in args.metrics:

        # Calculate the micro-MR1
        start_time = time.time()

        print("\nCalculating Micro-Averaged R1...")
        micro_mR1 = metrics.instance_based_MR1(embeddings, df)
        # Export the MR1 to txt
        output_path = os.path.join(output_dir, "micro_MR1.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(micro_mR1))

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Micro-Averaged R1: {micro_mR1:.3f} | Time: {time_str}")
        print(f"Results are exported to {output_path}")

    # Calculate Macro Averaged R1 (MR1) if required
    if "macro_mr1" in args.metrics:

        start_time = time.time()

        # Calculate MR1 for each label
        print("\nCalculating MR1 for each label...")
        label_mr1s, columns = macro_mR1 = metrics.calculate_mr1_for_labels(embeddings, df)
        print("MR1 for Top 5 labels:")
        for label, val, _ in label_mr1s[:5]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.3f}")
        print("MR1 for Bottom 5 labels:")
        for label, val, _ in label_mr1s[-5:]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.3f}")

        # Convert to a dataframe
        _df = pd.DataFrame(label_mr1s, columns=columns)
        # Export the MR1s to CSV
        output_path = os.path.join(output_dir, "labels_MR1.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to{output_path}")

        # Calculate the Balanced MR1
        print("\nCalculating the Balanced MR1...")
        balanced_mr1 = metrics.label_based_mr1(label_mr1s)
        # Export the MR1 to txt
        output_path = os.path.join(output_dir, "balanced_MR1.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(balanced_mr1))
        print(f"Balanced MR1: {balanced_mr1:.3f}")
        print(f"Results are exported to {output_path}")

        # Calculate the Family-based MR1
        # Read the family information
        with open(args.families_json, "r") as infile:
            families = json.load(infile)
        print("\nCalculating the Family based MR1...")
        family_mr1, columns = metrics.family_based_mr1(label_mr1s, families)
        # Convert to a dataframe
        _df = pd.DataFrame(family_mr1, columns=columns)
        # Export the labels' MR1s to CSV
        output_path = os.path.join(output_dir, "families_MR1.csv")
        _df.to_csv(output_path, index=False)
        print("    MR1 for each family:")
        for family, val in family_mr1:
            print(f"{family:>{len('Source-ambiguous_sounds')}}: {val:.3f}")
        print(f"Results are exported to{output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Time: {time_str}")

    #############
    print("Done!")
