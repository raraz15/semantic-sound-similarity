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

    # Calculate mAP@k for various values
    print("Calculating mAP@K for various k values...")
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
        total_time = time.time()-start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        print(f"K: {k:>{len(str(N))}} | mAP: {map:.5f} | Time: {time_str}")
    maps = pd.DataFrame(maps)

    # Export the mAP values
    dataset_name = os.path.basename(os.path.dirname(args.path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.path)))
    output_dir = os.path.join(EVAL_DIR, model_name,dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    results_name = os.path.splitext(os.path.basename(args.path))[0]
    output_path = os.path.join(output_dir, f"{results_name}.csv")
    print(f"Results are exported to {output_path}")
    maps.to_csv(output_path, index=False)

    #############
    print("Done!")