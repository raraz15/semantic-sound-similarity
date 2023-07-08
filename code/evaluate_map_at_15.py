""" Compute evaluation metrics for the similarity search result of an embedding 
over FSD50K.eval_audio."""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore") # Ignore 0 precision warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

import lib.metrics as metrics
from lib.directories import GT_PATH, EVAL_DIR, TAXONOMY_FAMILY_JSON

METRICS = ["micro_map@15", "macro_map@15"]

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_path',
                        type=str,
                        help='Path to similarity_results.json file.')
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
    if "micro_map@15" in args.metrics:

        start_time = time.time()

        # Calculate mAP@k for k=15
        print("\nCalculating Micro-Averaged mAP@15...")
        micro_map_at_15 = metrics.instance_based_map_at_n(results_dict, df, n=15)
        # Export the micro mAP@15 to txt
        output_path = os.path.join(output_dir, "micro_mAP@15.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(micro_map_at_15))

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Micro-Averaged mAP@15: {micro_map_at_15:.5f} | Time: {time_str}")
        print(f"Results are exported to {output_path}")

    # Calculate Macro Averaged Precision@15 (mAP@15) if required
    if "macro_map@15" in args.metrics:

        start_time = time.time()

        # Calculate mAP for each label
        print("\nCalculating mAP@15 for each label ...")
        label_maps, columns = metrics.calculate_map_at_n_for_labels(results_dict, df, n=15)
        print("mAP@15 for Top 5 labels:")
        for label, val in label_maps[:5]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.5f}")
        print("mAP@15 for Bottom 5 labels:")
        for label, val in label_maps[-5:]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.5f}")

        # Convert to a dataframe
        _df = pd.DataFrame(label_maps, columns=columns)
        # Export the mAP@15s to CSV
        output_path = os.path.join(output_dir, "labels_mAP@15.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to{output_path}")

        # Calculate the Balanced mAP@15, "AP computed on per-class basis, then averaged 
        # with equal weight across all classes to yield the overall performance"
        print("\nCalculating the Balanced mAP@15...")
        balanced_map_at_15 = metrics.label_based_map_at_n(label_maps)
        # Export the balanced mAP@15 to txt
        output_path = os.path.join(output_dir, "balanced_mAP@15.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(balanced_map_at_15))
        print(f"Balanced mAP@15: {balanced_map_at_15:.5f}")
        print(f"Results are exported to {output_path}")

        # Calculate the Family-based mAP@15
        # Read the family information
        with open(args.families_json, "r") as infile:
            families = json.load(infile)
        print("\nCalculating the Family-based mAP@15...")
        family_maps, columns = metrics.family_based_map_at_n(label_maps, families)
        # Convert to a dataframe
        _df = pd.DataFrame(family_maps, columns=columns)
        # Export the labels' maps to CSV
        output_path = os.path.join(output_dir, "families_mAP@15.csv")
        _df.to_csv(output_path, index=False)
        print(" mAP@15 for each family:")
        for family, val in family_maps:
            print(f"{family:>{len('Source-ambiguous_sounds')}}: {val:.5f}")
        print(f"Results are exported to{output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Time: {time_str}")

    #############
    print("Done!")
