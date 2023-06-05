""" Compute evaluation metrics for the similarity search result 
of an embedding over FSD50K.eval_audio."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

from directories import GT_PATH, EVAL_DIR

####################################################################################
# Utilities

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""

    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

####################################################################################
# Precision Related Metrics

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

def test_average_precision():
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
            import sys
            sys.exit(1)

def evaluate_relevance(query_fname, result, df, label=None):
    """ Evaluates the relevance of a result for a query. By default, A result is considered
    relevant if it has at least one label in common with the query. If a label is provided,
    a result is considered relevant if it contains the label."""

    # Get the labels of the query
    query_labels = get_labels(query_fname, df)
    # Evaluate the relevance of each retrieved document
    relevance = []
    for ref_result in result:
        ref_fname = ref_result["result_fname"]
        ref_labels = get_labels(ref_fname, df)
        if label is None:
            # Find if the retrieved element is relevant
            if len(query_labels.intersection(ref_labels)) > 0:
                relevance.append(1)
            else:
                relevance.append(0)
        else:
            # Find if the retrieved element contains the label
            if label in ref_labels:
                relevance.append(1)
            else:
                relevance.append(0)
    return relevance

def calculate_map_at_k(results_dict, df, k):
    """ Calculates the mean average precision (map) for the whole dataset. That is,
    each element in the dataset is considered a query and the average precision@k
    is calculated for each. Then, the mean of all these values is returned."""

    aps = [] # Average precision for the current k
    for query_fname, result in results_dict.items():
        # Evaluate the relevance of the result
        relevance = evaluate_relevance(query_fname, result[:k], df) # Cutoff at k
        # Calculate the average precision
        ap = average_precision(relevance)
        aps.append(ap)
    # mean average precision for the whole dataset
    map_at_k = sum(aps)/len(aps)
    return map_at_k

# TODO: is this the correct way of doing?
def calculate_micro_and_macro_averaged_precision_at_k(results_dict, df, k=15):
    """ Calculates the micro and macro mean averaged precision for the whole dataset. 
    That is, for each label in the dataset, the elements containing that label are 
    considered as queries and the true positive and false positive rates are calculated 
    for the result set. Then, all the true positives is divided by the sum of true 
    positives and false positives. """

    # Get all the labels from the df
    labels = set([label for labels in df["labels"].apply(lambda x: x.split(",")).to_list() for label in labels])
    # Calculate true positives and false positives for each label
    label_rates = []
    for label in list(labels):
        # Get the fnames containing this label
        fnames_with_label = df[df["labels"].apply(lambda x: label in x)]["fname"].to_list()
        # For each fname containing the label, calculate the total tp and fp
        tps, fps = [], []
        for query_fname in fnames_with_label:
            result = results_dict[str(query_fname)][:k] # Cutoff at k
            # Evaluate the relevance of the result
            relevance = evaluate_relevance(query_fname, result, df, label=label)
            # Calculate the true positive and false positive rates from the relevance
            tp = sum(relevance)
            fp = len(relevance)-tp
            tps.append(tp)
            fps.append(fp)
        # Append the tp and fp rates
        label_rates.append((sum(tps), sum(fps)))
    # Calculate the micro and macro averaged precision
    micro_ap = sum([tp for tp,_ in label_rates]) / (sum([tp+fp for tp,fp in label_rates]))
    macro_ap = sum([tp/(tp+fp) for tp,fp in label_rates]) / len(label_rates)
    return micro_ap, macro_ap

####################################################################################
# Ranking Related Metrics

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

####################################################################################
# Main

# TODO: ncdg
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('--increment', type=int, default=15, 
                        help="MAP@k calculation increments.")
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=["micro_ap", "macro_ap", "map", "mr1"],
                        help='Metrics to calculate.')
    args=parser.parse_args()
    args.metrics = [metric.lower() for metric in args.metrics] # Lowercase the metrics

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

    # Calculate mAP@k if required
    if "map" in args.metrics:
        print("Calculating mAP@k for various k values...")
        map_at_ks = []
        for k in range(args.increment, ((N//args.increment)+1)*args.increment, args.increment):
            start_time = time.time()
            map_at_k = calculate_map_at_k(results_dict, df, k)
            map_at_ks.append({"k": k, "mAP": map_at_k})
            time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
            print(f"k: {k:>{len(str(N))}} | mAP@k: {map_at_k:.5f} | Time: {time_str}")
        # Export the mAPs to CSV
        map_at_ks = pd.DataFrame(map_at_ks)
        output_path = os.path.join(output_dir, "mAP.csv")
        map_at_ks.to_csv(output_path, index=False)
        print(f"mAP@k results are exported to {output_path}")

    # Calculate Micro and Macro Averaged Precision@15 if required
    if "micro_ap" in args.metrics or "macro_ap" in args.metrics:
        print("Calculating Micro and Macro Averaged Precision@15...")
        start_time = time.time()
        micro_ap_at_k, macro_ap_at_k = calculate_micro_and_macro_averaged_precision_at_k(results_dict, 
                                                                                            df, 
                                                                                            k=15)
        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Micro Averaged Precision@15: {micro_ap_at_k:.5f} \
              | Macro Averaged Precision@15: {macro_ap_at_k:.5f}")
        print(f"Time: {time_str}")
        # Export the mAPs to txt if required
        if "micro_ap" in args.metrics:
            output_path = os.path.join(output_dir, "micro_averaged_precision_at_15.txt")
            with open(output_path, "w") as outfile:
                outfile.write(str(micro_ap_at_k))
                print(f"Results are exported to {output_path}")
        if "macro_ap" in args.metrics:
            output_path = os.path.join(output_dir, "macro_averaged_precision_at_15.txt")
            with open(output_path, "w") as outfile:
                outfile.write(str(macro_ap_at_k))
                print(f"Results are exported to {output_path}")

    # Calculate MR1 if requested
    if "mr1" in args.metrics:
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