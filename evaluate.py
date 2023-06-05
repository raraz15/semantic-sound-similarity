"""Compute evaluation metrics for the similarity search result 
of an embedding model over a dataset."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

from directories import GT_PATH, EVAL_DIR

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""

    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

def precision_at_k(y_true, k):
    """Calculate precision@k where k is in range(0,len(y_true))."""

    return sum(y_true[:k+1])/(k+1)

def average_precision(relevance):
    """Calculate the average prediction for a list of relevance values.
    relevance: list of relevance values (1: relevant, 0: not relevant)"""

    # Number of relevant documents
    total_relevant = sum(relevance)
    # If there are no relevant documents, return 0
    if total_relevant==0:
        return 0
    else:
        # Calculate average precision
        return sum([rel_k*precision_at_k(relevance,k) for k,rel_k in enumerate(relevance)]) / total_relevant

def test_ap():
    """Test the average precision function."""

    results = [
            [[0,0,0,0,0,0], 0.0],
            [[1,1,0,0,0,0], 1.0],
            [[0,0,0,0,1,1], 0.266],
            [[0,1,0,1,0,0], 0.5],
            ]
    for result,answer in results:
        delta = average_precision(result)-answer
        if abs(delta)>0.001:
            print("Error")

def calculate_average_precision(query_fname, result, df):
    """Calculates the average precision@k for a query and its results, where
    k is the length of the results. A result is considered relevant if it has
    at least one label in common with the query."""

    # Get the labels of the query
    query_labels = get_labels(query_fname, df)
    # Evaluate the relevance of each retrieved document
    relevance = []
    for ref_result in result:
        ref_fname = ref_result["result_fname"]
        ref_labels = get_labels(ref_fname, df)
        # Find if a retrieved element is relevant
        if len(query_labels.intersection(ref_labels)) > 0:
            relevance.append(1)
        else:
            relevance.append(0)
    # Calculate the average prediction
    return average_precision(relevance)

def calculate_map_at_k(results_dict, df, k):
    """Calculates the mean average precision for the whole dataset."""

    aps = [] # Average precision for the current k
    for query_fname, result in results_dict.items():
        aps.append(calculate_average_precision(query_fname, 
                                                result[:k], # Cutoff at k
                                                df)
                                                )
    map_at_k = sum(aps)/len(aps) # mean average precision for the whole dataset
    return map_at_k

def label_based_map(results_dict, df, k=15):
    """Calculates the average precision@k for each label in the query_fname."""

    # Get all the labels from the df
    labels = set([label for labels in df["labels"].apply(lambda x: x.split(",")).to_list() for label in labels])
    # Calculate the average precision for each label
    label_maps = []
    for label in list(labels):
        # Get the fnames containing this label
        fnames_with_label = df[df["labels"].apply(lambda x: label in x)]["fname"].to_list()
        # Calculate the ap for each fname
        aps = []
        for query_fname in fnames_with_label:
            result = results_dict[str(query_fname)][:k] # Cutoff at k
            ap = calculate_average_precision(query_fname, result, df)
            aps.append(ap)
        # mAP@k
        map = sum(aps)/len(aps)
        # Append the label and its map
        label_maps.append((label, map))
    # Sort the label maps by the mAP
    label_maps = sorted(label_maps, key=lambda x: x[1], reverse=True)
    # Convert to list of dicts
    label_maps = [{"label": label, "mAP": map, "k": k} for label,map in label_maps]

    return label_maps

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

# TODO: micro AP
# TODO: ncdg
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('-M', type=int, default=15, 
                        help="MAP@k calculation increments.")
    args=parser.parse_args()

    # Test the average precision function
    test_ap()

    # Read the ground truth annotations
    df = pd.read_csv(GT_PATH)

    # Read the results
    results_dict = {}
    with open(args.path, "r") as infile:
        for jline in infile:
            result_dict = json.loads(jline)
            results_dict[result_dict["query_fname"]] = result_dict["results"]
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
        map_at_k = calculate_map_at_k(results_dict, df, k)
        maps.append({"k": k, "mAP": map_at_k})
        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"k: {k:>{len(str(N))}} | mAP@k: {map_at_k:.5f} | Time: {time_str}")
    # Export the mAPs to CSV
    maps = pd.DataFrame(maps)
    output_path = os.path.join(output_dir, "mAP.csv")
    maps.to_csv(output_path, index=False)
    print(f"mAP@k results are exported to {output_path}")

    # Calculate label-based mAP@15
    print("Calculating label-based mAP@15...")
    label_based_map_at_15 = label_based_map(results_dict, df, k=15)
    # Calculate the mean label-based mAP@15
    av_label_based_map_at_15 = sum(map(lambda x: x["mAP"], label_based_map_at_15))/len(label_based_map_at_15)
    output_path = os.path.join(output_dir, "av_label_based_mAP_at_15.txt")
    with open(output_path, "w") as outfile:
        outfile.write(str(av_label_based_map_at_15))
    # Export the label-based mAP@15 to CSV
    label_based_map_at_15 = pd.DataFrame(label_based_map_at_15)
    output_path = os.path.join(output_dir, "label_based_mAP_at_15.csv")
    label_based_map_at_15.to_csv(output_path, index=False)
    print(f"label-based mAP@15 results are exported to {output_path}")

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