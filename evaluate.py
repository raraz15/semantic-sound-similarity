""" Compute evaluation metrics for the similarity search result 
of an embedding over FSD50K.eval_audio."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

import metrics
from directories import GT_PATH, EVAL_DIR

# TODO: ncdg
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('--increment', type=int, default=15, 
                        help="MAP@k calculation increments.")
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=["micro_ap", "macro_ap", "weighted_macro_ap", "map", "mr1"],
                        help='Metrics to calculate.')
    args=parser.parse_args()
    args.metrics = [metric.lower() for metric in args.metrics] # Lowercase the metrics

    # Test the average precision function
    metrics.test_average_precision()

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
        print("\nCalculating mAP@k for various k values...")
        map_at_ks = []
        for k in range(args.increment, ((N//args.increment)+1)*args.increment, args.increment):
            start_time = time.time()
            map_at_k = metrics.calculate_map_at_k(results_dict, df, k)
            map_at_ks.append({"k": k, "mAP": map_at_k})
            time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
            print(f"k: {k:>{len(str(N))}} | mAP@k: {map_at_k:.5f} | Time: {time_str}")
        # Export the mAPs to CSV
        map_at_ks = pd.DataFrame(map_at_ks)
        output_path = os.path.join(output_dir, "mAP.csv")
        map_at_ks.to_csv(output_path, index=False)
        print(f"mAP@k results are exported to {output_path}")

    # Calculate Micro, Macro or Weighted Macro Averaged Precision@15 if required
    if "micro_ap" in args.metrics or "macro_ap" in args.metrics or "weighted_macro_ap" in args.metrics:
        print("\nCalculating the number of True and False Positives ...")
        start_time = time.time()
        # Calculate positive rates for each label
        label_positive_rates = metrics.evaluate_label_positive_rates(results_dict, df, k=15)
        # Export the label positive rates to CSV
        _df = pd.DataFrame(label_positive_rates, columns=["tp", "fp", "label"])
        output_path = os.path.join(output_dir, "label_positive_rates.csv")
        _df.to_csv(output_path, index=False)
        print(f"Label positive rates are exported to {output_path}")
        # Remove the labels from the label positive rates
        label_positive_rates = [(tp,fp) for tp,fp,_ in label_positive_rates]
        # Calculate the micro averaged precision if required
        if "micro_ap" in args.metrics:
            print("Calculating the micro averaged precision ...")
            micro_averaged_precision = metrics.calculate_micro_averaged_precision(label_positive_rates)
            print(f"Micro Averaged Precision@15: {micro_averaged_precision:.5f}")
            output_path = os.path.join(output_dir, "micro_averaged_precision_at_15.txt")
            with open(output_path, "w") as outfile:
                outfile.write(str(micro_averaged_precision))
                print(f"Results are exported to {output_path}")
        # Calculate the macro averaged precision if required
        if "macro_ap" in args.metrics:
            print("Calculating the macro averaged precision ...")
            macro_averaged_precision = metrics.calculate_macro_averaged_precision(label_positive_rates)
            print(f"Macro Averaged Precision@15: {macro_averaged_precision:.5f}")
            output_path = os.path.join(output_dir, "macro_averaged_precision_at_15.txt")
            with open(output_path, "w") as outfile:
                outfile.write(str(macro_averaged_precision))
                print(f"Results are exported to {output_path}")
        if "weighted_macro_ap" in args.metrics:
            print("Calculating the weighted macro averaged precision ...")
            w_macro_averaged_precision = metrics.calculate_weighted_macro_averaged_precision(label_positive_rates)
            print(f"Weighted Macro Averaged Precision@15: {w_macro_averaged_precision:.5f}")
            output_path = os.path.join(output_dir, "weighted_macro_averaged_precision_at_15.txt")
            with open(output_path, "w") as outfile:
                outfile.write(str(w_macro_averaged_precision))
                print(f"Results are exported to {output_path}")
        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Time: {time_str}")

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
                mr1.append(metrics.R1(query_fname, result, df))
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