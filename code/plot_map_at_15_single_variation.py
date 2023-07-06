""" Takes a model variation and search anem and plots the evaluation results for it.
"""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from lib.plotting import plot_label_based_map_at_15

from lib.directories import FIGURES_DIR, EVAL_DIR

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', 
                        type=str, 
                        help='Name of the model. For example: '
                        'audioset-yamnet-1 or fs-essentia-extractor_legacy')
    parser.add_argument('variation', 
                        type=str, 
                        help='Name of the aggregation variation. For example: '
                        'Agg_mean-PCA_200-Norm_True')
    parser.add_argument('search',
                        type=str,
                        help='Name of the similarity search. For example: '
                        'nn or dot')
    parser.add_argument("--save-dir",
                        type=str,
                        default="",
                        help="Directory to save the figures.")
    parser.add_argument("--dataset",
                        type=str,
                        default="FSD50K.eval_audio",
                        help="Dataset used for evaluating.")
    args=parser.parse_args()

    # Create the save directory if it does not exist
    if args.save_dir == "":
        args.save_dir = os.path.join(FIGURES_DIR, args.model)

    # Plot the figures
    plot_label_based_map_at_15((args.model, args.variation, args.search),
                                EVAL_DIR,
                                dataset_name=args.dataset,
                                save_fig=True,
                                save_dir=args.save_dir)
    print("Done!")
