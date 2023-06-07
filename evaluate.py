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
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Path to results.json file.')
    parser.add_argument('--increment', type=int, default=15, 
                        help="MAP@k calculation increments.")
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=METRICS, 
                        help='Metrics to calculate.')
    args=parser.parse_args()

    # Check the metrics
    args.metrics = [metric.lower() for metric in args.metrics]
    assert set(args.metrics).issubset(set(METRICS)), \
        f"Invalid metrics. Valid metrics are: {METRICS}"

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

    # Calculate micro_mAP@k if required
    if "micro_map" in args.metrics:

        # Calculate mAP@k for k=15, 30, 45, ...
        print("\nCalculating micro mAP@k for various k values...")
        map_at_ks = []
        for k in range(args.increment, ((N//args.increment)+1)*args.increment, args.increment):
            start_time = time.time()
            micro_map_at_k = metrics.calculate_micro_map_at_k(results_dict, df, k)
            map_at_ks.append({"k": k, "mAP": micro_map_at_k})
            time_str = time.strftime('%M:%S', time.gmtime(time.time()-start_time))
            print(f"k: {k:>{len(str(N))}} | mAP@k: {micro_map_at_k:.5f} | Time: {time_str}")

        # Export the mAPs to CSV
        map_at_ks = pd.DataFrame(map_at_ks)
        output_path = os.path.join(output_dir, "micro_mAP.csv")
        map_at_ks.to_csv(output_path, index=False)
        print(f"Results are exported to {output_path}")

    # Calculate Macro or Weighted Macro Averaged Precision@15 if required
    if "macro_map" in args.metrics or "weighted_macro_map" in args.metrics:

        start_time = time.time()

        # Calculate mAP for each label
        print("\nCalculating macro mAP@15 for each label ...")
        label_maps, columns = metrics.calculate_map_at_k_for_labels(results_dict, df, k=15)
        # Convert to a dataframe
        _df = pd.DataFrame(label_maps, columns=columns)
        # Export the label positive rates to CSV
        output_path = os.path.join(output_dir, "labels_mAP@15.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to{output_path}")

        # Calculate the macro mAP@15 and weighted macro mAP@15
        print("\nCalculating the macro mAP@15 and weighted macro mAP@15...")
        macro_averaged_precision = metrics.calculate_macro_map(label_maps)
        w_macro_averaged_precision = metrics.calculate_weighted_macro_map(label_maps)
        print(f"Macro mAP@15: {macro_averaged_precision:.5f} |\
               Weighted Macro Averaged Precision@15: {w_macro_averaged_precision:.5f}")
        # Convert to a dataframe
        _df = pd.DataFrame([{"macro_map@15": macro_averaged_precision,
                            "weighted_macro_map@15": w_macro_averaged_precision}])
        # Export the results
        output_path = os.path.join(output_dir, "macro_mAP@15.csv")
        _df.to_csv(output_path, index=False)
        print(f"Results are exported to {output_path}")

        print(f"Time: {time.strftime('%M:%S', time.gmtime(time.time()-start_time))}")

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