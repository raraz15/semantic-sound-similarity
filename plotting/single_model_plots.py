"""Contains functions for plotting single model performance."""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

###################################################################################################
# Micro-averaged map@k

def plot_micro_map_at_15_comparisons(model, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and for each variation inside eval_dir,
    plots all the the micro-averaged AP@15 values in a single plot ."""

    default_fig_name = "Embedding processing and Search Algorithm Performances by "+ \
                f"Instance-Averaged AP@15 Values\n{model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read all the maps
    map_dict = {search: [] for search in searches}
    for search in searches:
        for model_dir in variation_paths:
            map_path = os.path.join(model_dir, search, "micro_mAP@15.txt")
            with open(map_path, "r") as in_f:
                micro_map_at_15 = float(in_f.read())
            full_model_name = model_dir.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            map_dict[search].append((variation, micro_map_at_15))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.25, 0.25, len(searches))
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
                            edgecolor='k')
            ax.text(z+positions[j], 
                            map+0.01, 
                            f"{map:.3f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=10)

    ax.set_title(f"Page 1 Results", fontsize=17)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.grid()
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Processing Parameters", fontsize=15)
    ax.legend(fontsize=10, loc=1, title="Search Algorithms", 
                        title_fontsize=10, fancybox=True)
    ax.set_ylim([0,1])
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-micro_mAP@15-comparisons.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

####################################################################################################
# Macro-averaged mAP@k

def plot_macro_map_at_15_comparisons(model, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and for each model variation inside eval_dir, 
    plots the Class-averaged mAP@15 in a single plot."""

    default_fig_name = "Embedding Processing and Search Algorithm Performances by "+\
                f"Label-Based mAP@15\n{model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read all the maps
    map_dict = {search: [] for search in searches}
    for search in searches:
        for variation_path in variation_paths:
            map_path = os.path.join(variation_path, search, "balanced_mAP@15.txt")
            with open(map_path, "r") as in_f:
                balanced_map_at_15 = float(in_f.read())
            full_model_name = variation_path.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            map_dict[search].append((variation, balanced_map_at_15))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.25, 0.25, len(searches))
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
                            edgecolor='k')
            ax.text(z+positions[j], 
                            map+0.01, 
                            f"{map:.3f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=10)

    ax.set_title(f"Page 1 Results", fontsize=17)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.grid()
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Processing Parameters", fontsize=15)
    ax.legend(fontsize=10, loc=1, title="Search Algorithms", 
                        title_fontsize=10, fancybox=True)
    ax.set_ylim([0,1])
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-macro_map@15-comparisons.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_label_based_map_at_15(model, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and finds all its variations in eval_dir. For each variation,
    plots the mAP@15 of the labels."""

    default_fig_name = f"Label-Based mAP@15 Values for {model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # For each variation of the model and the search path
    for variation_path in variation_paths:
        for search in searches:

            # Read the label-based mAP@15
            in_path = os.path.join(variation_path, search, "labels_mAP@15.csv")
            labels_map = pd.read_csv(in_path)

            # Get the labels and maps
            labels = labels_map["label"].to_list()
            maps = labels_map["map@15"].to_list()
            # Sort the labels and maps
            label_aps = sorted(zip(labels, maps), key=lambda x: x[1], reverse=True)

            # Determine some plot parameters
            N = 10 # Number of rows
            delta = len(maps) // N
            embedding_search = os.path.basename(variation_path).replace(model+"-", "") + "-" + search
            fig_name = fig_name if fig_name else default_fig_name

            # Plot the label-based mAP@15
            fig, ax = plt.subplots(figsize=(18, 24), nrows=N, constrained_layout=True)
            fig.suptitle(fig_name, fontsize=16)
            for i in range(N):
                ax[i].bar(
                        [label.replace("_","\n") for label,_ in label_aps[i*delta:(i+1)*delta]], 
                        [prec for _,prec in label_aps[i*delta:(i+1)*delta]]
                        )
                ax[i].set_yticks(np.arange(0, 1.05, 0.2))
                ax[i].grid()
                ax[i].set_ylim([0, 1.05])
                ax[i].set_ylabel("mAP@15")
            if save_fig:
                if save_dir == "":
                    print("Please provide a save directory if you want to save the figure.")
                    sys.exit(1)
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"{embedding_search}-label_based_mAP@15.png")
                print(f"Saving figure to {fig_path}")
                fig.savefig(fig_path)
                plt.close()

###################################################################################################
# MR1

def plot_mr1(model, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and plots the MR1 for all the variations of the model."""

    default_fig_name = "Embedding Processing and Search Algorithm " +\
                f"Performances by MR1 Values\n{model} Evaluated on {dataset_name}"

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
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
                edgecolor='k')
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
    ax.grid()
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-MR1.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()
