"""Compute evaluation metrics for the similarity search result of an embedding model over a dataset."""

import os
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings

import pandas as pd

from sklearn.metrics import average_precision_score

GT_PATH = "/data/FSD50K/FSD50K.ground_truth/eval.csv"
EVAL_DIR = "data/evaluation_results"

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""
    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

# TODO: remove counts
def calculate_average_precision(query_labels, result, df):
    """We define a retrieved document relevant when there is
    at least a match."""
    y_true, y_score, counts = [], [], []
    for ref_result in result:
        ref_fname = list(ref_result.keys())[0]
        ref_score = list(ref_result.values())[0]
        y_score.append(ref_score)
        ref_labels = get_labels(ref_fname, df)
        # Find how many labels are shared
        counter = len(query_labels.intersection(ref_labels))
        counts.append(counter)
        if counter > 0:
            y_true.append(1)
        else:
            y_true.append(0)
    # Calculate the average prediction
    ap = average_precision_score(y_true, y_score)
    return ap, counts

def R1(query_labels, result, df):
    for i,ref_result in enumerate(result):
        ref_fname = list(ref_result.keys())[0]
        ref_labels = get_labels(ref_fname, df)
        # Find where the first match is
        counter = len(query_labels.intersection(ref_labels))
        if counter > 0:
            return i

# TODO: MR1@K
# TODO: MR1 NaNs
# TODO: ncdg
if __name__=="__main__":

    parser=argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('-M', type=int, default=15, 
                        help="MAP@k calculation increments.")
    args=parser.parse_args()

    # Read the ground truth annotations
    df = pd.read_csv(GT_PATH)

    # Read the similarity search results
    with open(args.path, "r") as infile:
        results_dict = json.load(infile)
    fnames = list(results_dict.keys())
    N = len(results_dict[fnames[0]]) # Number of returned results for each query

    # Create the output directory
    search_name = os.path.basename(os.path.dirname(args.path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.path)))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.path))))
    output_dir = os.path.join(EVAL_DIR, dataset_name, model_name, search_name)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate mAP@k for various values
    print("Calculating mAP@K for various K values...")
    maps = []
    for k in range(args.M,((N//args.M)+1)*args.M,args.M):
        start_time = time.time()
        aps = []
        for query_fname in fnames:
            query_labels = get_labels(query_fname, df)
            result = results_dict[query_fname][:k] # Cutoff at k
            ap, _ = calculate_average_precision(query_labels, result, df)
            aps.append(ap)
        map = sum(aps)/len(aps) # mean average precision
        maps.append({"k": k, "mAP": map})
        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"K: {k:>{len(str(N))}} | mAP: {map:.5f} | Time: {time_str}")
    # Export
    maps = pd.DataFrame(maps)
    output_path = os.path.join(output_dir, "mAP.csv")
    maps.to_csv(output_path, index=False)
    print(f"mAP results are exported to {output_path}")

    # Calculate MR1
    print("\nCalculating MR1...")
    start_time = time.time()
    mr1 = []
    for query_fname in fnames:
        query_labels = get_labels(query_fname, df)
        result = results_dict[query_fname]
        mr1.append(R1(query_labels, result, df))
    mr1 = [x for x in mr1 if x] # Remove entries with no matches
    mr1 = sum(mr1)/len(mr1)
    time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
    print(f"MR1: {mr1:.1f} | Time: {time_str}")
    # Export
    output_path = os.path.join(output_dir, "MR1.txt")
    with open(output_path, "w") as outfile:
        outfile.write(str(mr1))
    print(f"MR1 results are exported to {output_path}")

    #############
    print("Done!")