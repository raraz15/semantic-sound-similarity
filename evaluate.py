"""Compute evaluation metrics for the similarity search result 
of an embedding model over a dataset."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

from sklearn.metrics import average_precision_score

from directories import GT_PATH, EVAL_DIR

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""
    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(", "))

# TODO: use your own method because nearest neighbors does not work with this score
def calculate_average_precision(query_fname, result, df):
    """We define a retrieved document relevant when there is
    at least a match."""
    query_labels = get_labels(query_fname, df)
    # Evaluate if each document is relevant
    y_true, y_score = [], []
    for ref_result in result:
        y_score.append(ref_result["score"])
        ref_labels = get_labels(ref_result["result_fname"], df)
        # Find how many labels are shared
        if len(query_labels.intersection(ref_labels)) > 0:
            y_true.append(1)
        else:
            y_true.append(0)
    # Calculate the average prediction
    ap = average_precision_score(y_true, y_score)
    return ap

# TODO: MR1 NaNs
def R1(query_fname, result, df):
    query_labels = get_labels(query_fname, df)
    for i,ref_result in enumerate(result):
        ref_fname = ref_result["result_fname"]
        ref_labels = get_labels(ref_fname, df)
        # Find if there is a match in the labels
        if len(query_labels.intersection(ref_labels)) > 0:
            return i # Return the rank of the first match
    return None # No match

# TODO: macro AP, micro AP
# TODO: ncdg
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('-M', type=int, default=15, 
                        help="MAP@k calculation increments.")
    args=parser.parse_args()

    # Read the ground truth annotations
    df = pd.read_csv(GT_PATH)

    # Read one similarity search result to get the N
    with open(args.path, "r") as infile:
        for jline in infile:
            result_dict = json.loads(jline)
            break
    N = len(result_dict["results"]) # Number of returned results for each query

    # Create the output directory
    search_name = os.path.basename(os.path.dirname(args.path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.path)))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.path))))
    output_dir = os.path.join(EVAL_DIR, dataset_name, model_name, search_name)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate mAP@k for various k values
    print("Calculating mAP@k for various k values...")
    maps = []
    for k in range(args.M, ((N//args.M)+1)*args.M, args.M):
        start_time = time.time()
        aps = [] # Average precision for the current k
        with open(args.path, "r") as infile:
            for jline in infile:
                result_dict = json.loads(jline)
                query_fname = result_dict["query_fname"]
                result = result_dict["results"][:k] # Cutoff at k
                aps.append(calculate_average_precision(query_fname, result, df))
        map_k = sum(aps)/len(aps) # mean average precision @k for the whole dataset
        maps.append({"k": k, "mAP": map_k})
        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"k: {k:>{len(str(N))}} | mAP@k: {map_k:.5f} | Time: {time_str}")
    # Export the mAPs to CSV
    maps = pd.DataFrame(maps)
    output_path = os.path.join(output_dir, "mAP.csv")
    maps.to_csv(output_path, index=False)
    print(f"mAP@k results are exported to {output_path}")

    # Calculate MR1
    print("\nCalculating MR1...")
    start_time = time.time()
    mr1 = []
    with open(args.path, "r") as infile:
        for jline in infile:
            results_dict = json.loads(jline)
            query_fname = results_dict["query_fname"]
            result = results_dict["results"]
            mr1.append(R1(query_fname, result, df))
    mr1 = [x for x in mr1 if x] # Remove entries with no matches
    mr1 = sum(mr1)/len(mr1)
    time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
    print(f"MR1: {mr1:.1f} | Time: {time_str}")
    # Export the MR1s to txt
    output_path = os.path.join(output_dir, "MR1.txt")
    with open(output_path, "w") as outfile:
        outfile.write(str(mr1))
    print(f"MR1 results are exported to {output_path}")

    #############
    print("Done!")