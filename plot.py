import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from plotting import plot_map_at_15, plot_label_based_p_at_15, plot_av_label_based_p_at_15, plot_mr1
from directories import FIGURES_DIR, EVAL_DIR

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', 
                        type=str, 
                        required=True, 
                        help='Name of the model.')
    parser.add_argument("--save-dir",
                        type=str,
                        default="",
                        help="Directory to save the figures.")
    parser.add_argument("--dataset",
                        type=str,
                        default="FSD50K.eval_audio",
                        help="Dataset used for evaluating.")
    args=parser.parse_args()

    if args.save_dir == "":
        args.save_dir = os.path.join(FIGURES_DIR, args.model)

    plot_map_at_15(args.model, EVAL_DIR, args.dataset, save_fig=True, save_dir=args.save_dir)
    plot_label_based_p_at_15(args.model, EVAL_DIR, args.dataset, save_fig=True, save_dir=args.save_dir)
    plot_av_label_based_p_at_15(args.model, EVAL_DIR, args.dataset, save_fig=True, save_dir=args.save_dir)
    plot_mr1(args.model, EVAL_DIR, args.dataset, save_fig=True, save_dir=args.save_dir)
    print("Done!")
