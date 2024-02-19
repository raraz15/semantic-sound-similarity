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
from lib.directories import EVAL_DIR

METRICS = ["micro_map@n", "macro_map@n"]

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_path',
                        type=str,
                        help='Path to similarity_results.json file.')
    parser.add_argument("ground_truth",
                        type=str,
                        help="Path to the ground truth CSV file. "
                        "You can provide a subset of the ground truth by "
                        "filtering the CSV file before passing it to this script.")
    parser.add_argument('--metrics',
                        type=str,
                        nargs='+',
                        default=METRICS, 
                        help='Metrics to calculate.')
    parser.add_argument('-N', 
                        type=int, 
                        default=15, 
                        help="Cutoff rank.")
    parser.add_argument("--families-json",
                        type=str,
                        default=None,
                        help="Path to the JSON file containing the family information "
                        "of the FSD50K Taxonomy. You can also provide the family "
                        "information from an ontology")
    parser.add_argument("--output-dir",
                        "-o",
                        type=str,
                        default=EVAL_DIR,
                        help="Path to the output directory.")
    args=parser.parse_args()

    # Check the metrics
    args.metrics = [metric.lower() for metric in args.metrics]
    assert set(args.metrics).issubset(set(METRICS)), \
        f"Invalid metrics. Valid metrics are: {METRICS}"

    # Read the ground truth annotations
    df = pd.read_csv(args.ground_truth)
    df["fname"] = df["fname"].astype(str) # To unify different datasets
    gt_fnames = set(df["fname"].to_list())
    print(f"Number of queries in the ground truth file: {len(gt_fnames):,}")

    # Read the results
    print("Reading the similarity results...")
    results_dict = {}
    with open(args.results_path, "r") as infile:
        for jline in infile:
            result_dict = json.loads(jline)
            query_fname = result_dict["query_fname"]
            # Only calculate metrics for queries that are in the ground truth
            if query_fname in gt_fnames:
                results_dict[query_fname] = result_dict["results"]
                assert args.N <= len(result_dict["results"]), \
                f"Number of returned results for {query_fname} is less than {args.N}."

    # Determine the output directory
    search_name = os.path.basename(os.path.dirname(args.results_path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(args.results_path)))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.results_path))))
    output_dir = os.path.join(args.output_dir, dataset_name, model_name, search_name)
    print(f"Output directory: {output_dir}")
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Micro-Averaged Average Precision@N if required
    if "micro_map@n" in args.metrics:

        start_time = time.time()

        # Calculate Instance-based mAP@N
        print(f"Calculating Micro-Averaged mAP@{args.N}...")
        micro_map_at_N = metrics.instance_based_map_at_n(results_dict, df, n=args.N)
        # Export the micro mAP@N to txt
        output_path = os.path.join(output_dir, f"micro_mAP@{args.N}.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(micro_map_at_N))

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print("-"*40)
        print(f"Micro-Averaged mAP@{args.N}: {micro_map_at_N:.5f} | Time: {time_str}")
        print("-"*40)
        print(f"Results are exported to {output_path}")

    # Calculate Macro-Averaged Average Precision@N if required
    if "macro_map@n" in args.metrics:

        start_time = time.time()

        # Calculate mAP for each label
        print(f"\nCalculating mAP@{args.N} for each label ...")
        label_maps, columns = metrics.calculate_map_at_n_for_labels(results_dict, df, n=args.N)
        print("-"*40)
        print(f"mAP@{args.N} for Top 5 labels")
        for label, val, _ in label_maps[:5]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.5f}")
        print(" "*15+"-"*10+" "*15)
        print(f" mAP@{args.N} for Bottom 5 labels")
        for label, val, _ in label_maps[-5:]:
            print(f"{label:>{len('Source-ambiguous_sounds')}}: {val:.5f}")
        print("-"*40)

        # Convert to a dataframe
        _df = pd.DataFrame(label_maps, columns=columns)
        # Export each label-based mAP@N to CSV
        output_path = os.path.join(output_dir, f"labels_mAP@{args.N}.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to{output_path}")

        # Calculate the Balanced mAP@N, "AP computed on per-class basis, then averaged 
        # with equal weight across all classes to yield the overall performance"
        print(f"\nCalculating the Balanced mAP@{args.N}...")
        balanced_map_at_N = metrics.label_based_map_at_n(label_maps)
        # Export the balanced mAP@N to txt
        output_path = os.path.join(output_dir, f"balanced_mAP@{args.N}.txt")
        with open(output_path, "w") as outfile:
            outfile.write(str(balanced_map_at_N))
        print("-"*40)
        print(f"Balanced mAP@{args.N}: {balanced_map_at_N:.5f}")
        print("-"*40)
        print(f"Results are exported to {output_path}")

        if args.families_json is not None:
            # Read the family information
            with open(args.families_json, "r") as infile:
                families = json.load(infile)
            print(f"\nCalculating the Family-based mAP@{args.N}...")
            family_maps, columns = metrics.family_based_map_at_n(label_maps, families)
            # Convert to a dataframe
            _df = pd.DataFrame(family_maps, columns=columns)
            # Export the labels' maps to CSV
            output_path = os.path.join(output_dir, f"families_mAP@{args.N}.csv")
            _df.to_csv(output_path, index=False)
            print("-"*40)
            print(f"  mAP@{args.N} for each family")
            for family, val in family_maps:
                print(f"{family:>{len('Source-ambiguous_sounds')}}: {val:.5f}")
            print("-"*40)
            print(f"Results are exported to {output_path}")

        time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
        print(f"Time: {time_str}")

    #############
    print("Done!\n")
