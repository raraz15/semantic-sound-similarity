"""Contains functions for plotting each variation of a model on the same figure."""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

from .utils import save_function, sort_variation_paths, get_pca
from ..directories import EVAL_DIR, DATASET_NAME

def plot_map_at_15_comparisons(model, map_type,
                                eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                fig_name="", save_fig=False, save_dir="",
                                presentation_mode=False):
    """Takes a model name and for each variation inside eval_dir,
    plots all the the micro-averaged AP@15 values in a single plot ."""

    # Determine the file name and figure name
    # TODO: Instead of Label Class?
    if map_type=="micro":
        file_name = "micro_mAP@15.txt"
        default_fig_name = f"Sound Similarity Performances of AIR Systems using " \
                            f"Instance-Based mAP@15\n{model} Evaluated on {dataset_name}"
        figure_save_name = "micro_mAP@15-comparisons.png"
    elif map_type=="macro":
        file_name = "balanced_mAP@15.txt"
        default_fig_name = f"Sound Similarity Performances of AIR Systems using " \
                            f"Label-Based mAP@15\n{model} Evaluated on {dataset_name}"
        figure_save_name = "macro_map@15-comparisons.png"
    else:
        raise("map_type must be one of 'micro', 'macro'")

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Sort further by PCA
    variation_paths = sort_variation_paths(model, variation_paths)
    # Deal with presentation mode
    if presentation_mode:
        _variation_paths = []
        for var_path in variation_paths:
            n_pca = int(var_path.split("/")[-1].split("-PCA_")[1].split("-Norm")[0])
            if n_pca>=100:
                _variation_paths.append(var_path)
            elif n_pca==64 and "vggish" in model:
                _variation_paths.append(var_path)
        variation_paths = _variation_paths
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read all the maps
    map_dict = {search: [] for search in searches}
    for search in searches:
        for model_dir in variation_paths:
            map_path = os.path.join(model_dir, search, file_name)
            with open(map_path, "r") as in_f:
                micro_map_at_15 = float(in_f.read())
            full_model_name = model_dir.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            map_dict[search].append((variation, micro_map_at_15))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.2, 0.2, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1
    fig_name = fig_name if fig_name else default_fig_name

    # Plot the maps
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    xticks = []
    for j,search in enumerate(map_dict.keys()):
        for z,(variation,map) in enumerate(map_dict[search]):
            if j==0:
                if "essentia" not in variation:
                    xticks.append(variation.replace("-","\n").replace("Agg_", ""))
                else:
                    xticks.append(variation.replace("essentia-extractor_legacy-", ""))
            if z==0:
                if search=="dot":
                    label = "Dot Product"
                elif search=="nn":
                    label = "Nearest Neighbors"
            else:
                label = ""
            ax.bar(z+positions[j], 
                    height=map, 
                    width=delta*0.6, 
                    label=label, 
                    color=COLORS[j], 
                    edgecolor='k',
                    linewidth=1.2)
            ax.text(z+positions[j]+0.05*(-1)**(j+1) , 
                    map+0.01, 
                    f"{map:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10)

    #ax.set_title(f"Page 1 Results", fontsize=17)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.set_ylim([0,1])
    ax.set_xlim([-0.75, len(xticks)-0.25])
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Embedding Processing Parameters", fontsize=15)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=10, loc=1, title="Search Algorithms", 
            title_fontsize=10, fancybox=True)

    save_function(save_fig, save_dir, figure_save_name, fig)
    plt.show()

def plot_macro_map_at_15_PCA_comparisons(model_search, 
                                         eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                         fig_name="", save_fig=False, save_dir=""):
    """ Takes a model name, a fixed aggregation, normalization, and fixed search type 
    and plots the map@15 of each model variation inside eval_dir following these parameters in the same
    plot.
    """

    model, agg, norm, search = model_search

    default_fig_name = f"Effect of the Number of PCA Components on Sound Similarity Performace by "+\
                f"Label-Averaged mAP@15\n{model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    if model=="fs-essentia-extractor_legacy":
        wildcard = f"{model}-PCA_*"
    else:
        wildcard = f"{model}-Agg_{agg}-PCA_*-Norm_{norm}"
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, wildcard)))
    # Sort by PCA components
    variation_paths = sort_variation_paths(model, variation_paths)

    # Read all the maps
    variations, maps = [], []
    for variation_path in variation_paths:
        map_path = os.path.join(variation_path, search, "balanced_mAP@15.txt")
        with open(map_path, "r") as in_f:
            balanced_map_at_15 = float(in_f.read())
        full_model_name = variation_path.split("/")[-1]
        if "fs-essentia-extractor_legacy" in full_model_name:
            variation = "-"+full_model_name.split("-")[-1]
        else:
            variation = "-".join(full_model_name.split("-")[-3:])
        maps.append(balanced_map_at_15)
        variations.append(variation)

    # Determine some plot parameters
    fig_name = fig_name if fig_name else default_fig_name

    # Plot the maps
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(default_fig_name, fontsize=19, weight='bold')

    xticks = []
    for i,(variation,balanced_mAP) in enumerate(zip(variations, maps)):
        ax.bar(i,
                height=balanced_mAP, 
                width=0.85,
                color=COLORS[0],
                edgecolor='k',
                linewidth=1.2)
        ax.text(i, 
                balanced_mAP+0.01, 
                f"{balanced_mAP:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=12)
        xticks.append(get_pca(variation))

    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.grid(alpha=0.5)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Number of PCA Components", fontsize=15)
    ax.set_ylim([0,1])

    save_function(save_fig, save_dir, f"{'-'.join(model_search)}-macro_map@15-PCA_comparisons.png", fig)
    plt.show()

# TODO: since we dont compute MR1 for all models, this function is not useful anymore
def plot_mr1(model, 
             eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
             fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and plots the MR1 for all the variations of the model."""

    default_fig_name = "Embedding Processing and Search Algorithm " +\
                f"Performances by MR1 Values\n{model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Sort further
    variation_paths = sort_variation_paths(model, variation_paths)
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read the MR1s
    mr1_dict = {}
    for search in searches:
        mr1_dict[search] = []
        for model_dir in variation_paths:
            mr1_path = os.path.join(model_dir, search, "MR1.txt")
            with open(mr1_path,"r") as infile:
                mr1 = float(infile.read())
            # Format the variation name for pprint
            full_model_name = model_dir.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            mr1_dict[search].append((variation, mr1))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.2, 0.2, len(searches))
    else:
        positions = [0]
    fig_name = fig_name if fig_name else default_fig_name
    # Plot the MR1s
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    xticks, max_val = [], []
    for i in range(len(variation_paths)):
        for j,search in enumerate(searches):
            variation, mr1 = mr1_dict[search][i]
            max_val += [mr1]
            if j%len(searches)==0:
                xticks.append(variation.replace("-","\n").replace("Agg_", ""))
            if i==0:
                if search=="dot":
                    label = "Dot Product"
                elif search=="nn":
                    label = "Nearest Neighbors"
            else:
                label = ""
            ax.bar(i+positions[j], 
                height=mr1, 
                width=0.35, 
                label=label, 
                color=COLORS[j], 
                edgecolor='k',
                linewidth=1.2)
            ax.text(i+positions[j], 
                    mr1+0.01, 
                    f"{mr1:.2f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10)

    # Set the plot parameters
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,max(max_val)+0.5,0.5))
     # TODO: read K
    ax.set_ylabel("MR1@90 (↓)", fontsize=15)
    ax.set_xlabel("Embedding Processing Parameters", fontsize=15)
    ax.legend(loc=4, fontsize=11, title="Search Algorithms", 
              title_fontsize=12, fancybox=True)
    ax.grid(alpha=0.5)

    save_function(save_fig, save_dir, "mr1-comparisons.png", fig)
    plt.show()
