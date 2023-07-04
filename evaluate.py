""" Compute evaluation metrics for the similarity search result of an embedding 
over FSD50K.eval_audio."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

import metrics
from directories import GT_PATH, EVAL_DIR

METRICS = ["micro_map", "macro_map", "mr1"]

# TODO: ncdg
# TODO: GAP@k
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_path',
                        type=str,
                        help='Path to results.json file.')
    parser.add_argument('--increment', 
                        type=int, 
                        default=15, 
                        help="MAP@k calculation increments.")
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
    parser.add_argument("--output-dir",
                        type=str,
                        default=EVAL_DIR,
                        help="Path to the output directory.")
    args=parser.parse_args()

    # Check the metrics
    args.metrics = [metric.lower() for metric in args.metrics]
    assert set(args.metrics).issubset(set(METRICS)), \
        f"Invalid metrics. Valid metrics are: {METRICS}"

    # Test the average precision function
    metrics.test_average_precision_at_n()

    # Read the ground truth annotations
    df = pd.read_csv(args.ground_truth)
    fnames = set(df["fname"].to_list())

    # Read the results
    results_dict = {}
    with open(args.results_path, "r") as infile:
        for jline in infile:
            result_dict = json.loads(jline)
            # Only calculate metrics for queries that are in the ground truth
            if int(result_dict["query_fname"]) in fnames:
                results_dict[result_dict["query_fname"]] = result_dict["results"]
    N = len(result_dict["results"]) # Number of returned results for each query

    # Determine the output directory
    search_name = os.path.basename(os.path.dirname(args.results_path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.results_path)))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.results_path))))
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, search_name)
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Micro-Averaged mAP@15 if required
    if "micro_map" in args.metrics:

        start_time = time.time()

        # Calculate mAP@k for k=15
        print("\nCalculating Micro-Averaged mAP@15...")
        micro_map_at_15 = metrics.instance_based_map_at_n(results_dict, df, n=15)

        # Export the micro mAP@15 to txt
        output_path = os.path.join(output_dir, "micro_mAP@15.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(micro_map_at_15))
        print(f"Results are exported to {output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Micro-Averaged mAP@15: {micro_map_at_15:.5f} | Time: {time_str}")

    # Calculate Macro Averaged Precision@15 if required
    if "macro_map" in args.metrics:

        start_time = time.time()

        # Calculate mAP for each label
        print("\nCalculating mAP@15 for each label ...")
        label_maps, columns = metrics.calculate_map_at_n_for_labels(results_dict, df, n=15)
        # Convert to a dataframe
        _df = pd.DataFrame(label_maps, columns=columns)
        # Export the labels' maps to CSV
        output_path = os.path.join(output_dir, "labels_mAP@15.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to{output_path}")

        # Calculate the Balanced mAP@15, "AP computed on per-class basis, then averaged 
        # with equal weight across all classes to yield the overall performance"
        print("\nCalculating the Balanced mAP@15...")
        balanced_map_at_15 = metrics.label_based_map_at_n(label_maps)

        # Export the micro mAP@15 to txt
        output_path = os.path.join(output_dir, "balanced_mAP@15.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(balanced_map_at_15))
        print(f"Results are exported to {output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Balanced mAP@15: {balanced_map_at_15:.5f} | Time: {time_str}")

    # Calculate MR1 if requested
    if "mr1" in args.metrics:

        start_time = time.time()

        # Calculate MR1 for each query
        print("\nCalculating MR1...")
        mr1 = metrics.calculate_MR1(results_dict, df)

        # Export the MR1s to txt
        output_path = os.path.join(output_dir, "MR1.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(mr1))
        print(f"Results are exported to {output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"MR1: {mr1:.1f} | Time: {time_str}")

    #############
    print("Done!")
