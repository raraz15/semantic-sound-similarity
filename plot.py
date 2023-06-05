import os
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from directories import FIGURES_DIR, EVAL_DIR
DATASET_NAME = "FSD50K.eval_audio"
EVAL_DIR = os.path.join(EVAL_DIR, DATASET_NAME)

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

#####################################################################################
# Utility Functions

def get_model_name(full_name):
    return full_name.split("-PCA")[0].split("-Agg")[0]

#####################################################################################
# Single Model Plots

def plot_map_at_all_k(model, eval_dir=EVAL_DIR, n_cols=3, 
                      save_fig=False, save_dir=FIGURES_DIR):
    """Takes a model name and plots the mAP@k for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))

    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])
    # Read one map to get the k values
    map_path = os.path.join(variation_paths[0], searches[0], "mAP.csv")
    k_values = pd.read_csv(map_path).k.to_list()

    # Read all the maps
    map_dict = {}
    for k in k_values:
        map_dict[k] = {search: [] for search in searches}
        for search in searches:
            for model_dir in variation_paths:
                map_path = os.path.join(model_dir, search, "mAP.csv")
                map = pd.read_csv(map_path)
                full_model_name = model_dir.split("/")[-1]
                variation = "-".join(full_model_name.split("-")[-3:])
                map_dict[k][search].append((variation, map[map.k==k].mAP.to_numpy()[0]))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.20, 0.20, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1
    n_rows = len(map_dict.keys())//n_cols

    # Plot the maps
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, 
                            figsize=(6*n_cols,6*n_rows), constrained_layout=True)
    fig.suptitle(f"Embedding Processing and Search Algorithm Performances by mAP Values"
                 f"\n{model} Evaluated on {DATASET_NAME}", fontsize=20, weight='bold')
    #fig.suptitle(f"{model} - mAP@k Values on {DATASET_NAME}", fontsize=20, weight='bold')
    for n,k in enumerate(map_dict.keys()):

        row, col = n//n_cols, n%n_cols
        xticks = []
        for j,search in enumerate(map_dict[k].keys()):
            for z,(variation,map) in enumerate(map_dict[k][search]):
                if j==0:
                    xticks.append(variation.replace("-","\n"))
                if z==0:
                    if search=="dot":
                        label = "Dot Product"
                    elif search=="nn":
                        label = "Nearest Neighbors"
                else:
                    label = ""
                axs[row,col].bar(z+positions[j], 
                                height=map, 
                                width=delta*0.8, 
                                label=label, 
                                color=COLORS[j], 
                                edgecolor='k')
                axs[row,col].text(z+positions[j], 
                                map+0.01, 
                                f"{map:.3f}", 
                                ha='center', 
                                va='bottom', 
                                fontsize=10)

        axs[row,col].set_title(f"k={k} (Page {k//15})", fontsize=17, weight='bold')
        axs[row,col].tick_params(axis='y', which='major', labelsize=11)
        axs[row,col].tick_params(axis='x', which='major', labelsize=10)
        axs[row,col].set_xticks(np.arange(len(xticks)), xticks) #, rotation=20
        axs[row,col].set_yticks(np.arange(0,1.05,0.05))
        axs[row,col].grid()
        axs[row,col].set_ylabel("mAP@k (↑)", fontsize=15)
        axs[row,col].set_xlabel("Processing Parameters", fontsize=15)
        axs[row,col].legend(fontsize=11, loc=4, title="Search Algorithms", 
                            title_fontsize=12, fancybox=True)
        axs[row,col].set_ylim([0,1])
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-mAP.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_map_at_15(model, eval_dir=EVAL_DIR, D=0.25, 
                   save_fig=False, save_dir=FIGURES_DIR):
    """Takes a model name and plots the mAP@k for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))

    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])
    # Read one map to get the k values
    map_path = os.path.join(variation_paths[0], searches[0], "mAP.csv")
    k_values = pd.read_csv(map_path).k.to_list()

    # Read all the maps
    map_dict = {}
    for k in k_values:
        if k!=15:
            continue
        map_dict[k] = {search: [] for search in searches}
        for search in searches:
            for model_dir in variation_paths:
                map_path = os.path.join(model_dir, search, "mAP.csv")
                map = pd.read_csv(map_path)
                full_model_name = model_dir.split("/")[-1]
                variation = "-".join(full_model_name.split("-")[-3:])
                map_dict[k][search].append((variation, map[map.k==k].mAP.to_numpy()[0]))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-D, D, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1

    # Plot the maps
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Processing and Search Algorithm Performances by mAP"
                 f"\n{model} Evaluated on {DATASET_NAME}", fontsize=20, weight='bold')
    k = 15
    xticks = []
    for j,search in enumerate(map_dict[k].keys()):
        for z,(variation,map) in enumerate(map_dict[k][search]):
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
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-mAP@15.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_mr1(model, eval_dir=EVAL_DIR, 
             save_fig=False, save_dir=FIGURES_DIR):
    """Takes a model name and plots the MR1 for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))

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

    # Plot the MR1s
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Processing and Search Algorithm Performances by MR1 Values", fontsize=20, weight='bold')
    ax.set_title(f"{model} Evaluated on {DATASET_NAME}", fontsize=18)
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
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-MR1.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_label_based_map(model, eval_dir=EVAL_DIR, 
                         save_fig=False, save_dir=FIGURES_DIR):
    """Takes an embedding name and plots the label-based mAP@15 for only that variation."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read the label-based mAP@15
    for variation_path in variation_paths:
        for search in searches:

            embedding_search = os.path.basename(variation_path) + "-" + search

            # Read the label-based mAP@15
            map_path = os.path.join(variation_path, search, "label_based_mAP_at_15.csv")
            label_maps = pd.read_csv(map_path)
            k = label_maps.k.to_list()[0] # All k values are the same # TODO: read K or fix?
            # Separate the labels and maps
            labels = label_maps.label.to_list()
            maps = label_maps.mAP.to_list()

            # Plot the label-based mAP@15
            N = 10 # Number of rows
            delta = len(label_maps) // N
            fig, ax = plt.subplots(figsize=(18, 24), nrows=N, constrained_layout=True)
            fig.suptitle(f"{embedding_search} mAP Values of Individual Classes", fontsize=16)
            for i in range(N):
                ax[i].bar([label.replace("_","\n") for label in labels[i*delta:(i+1)*delta]], 
                        [count for count in maps[i*delta:(i+1)*delta]])
                ax[i].set_yticks(np.arange(0, 1.1, 0.2))
                ax[i].grid()
                ax[i].set_ylim([0, 1.1])
                ax[i].set_ylabel("mAP@{}".format(k))
            if save_fig:
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"{embedding_search}-label_based_mAP_at_15.png")
                print(f"Saving figure to {fig_path}")
                fig.savefig(fig_path)
                plt.close()

#####################################################################################
# Multiple Model Plots

# TODO: check MR1@90
def plot_mr1_comparisons_single_variation(models, eval_dir=EVAL_DIR, 
                                          save_fig=False, save_dir=FIGURES_DIR):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Read the MR1s for each model
    mr1s = []
    for model in models:
        model_dir = os.path.join(eval_dir, model[0])
        results_dir = os.path.join(model_dir, model[1])
        mr1_path = os.path.join(results_dir, "MR1.txt")
        with open(mr1_path,"r") as infile:
            mr1 = float(infile.read())
        mr1s.append((model[0], model[1], mr1))

    # Plot the MR1s
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Performances using MR1 values Evaluated on {DATASET_NAME} Set", fontsize=20, weight='bold')
    ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    for i,(model_name,search,mr1) in enumerate(mr1s):
            ax.bar(i, 
                mr1, 
                label=get_model_name(model_name),
                width=1*0.85, 
                color=COLORS[i], 
                edgecolor='k'
                )
            ax.text(i, 
                mr1+0.01, 
                f"{mr1:.2f}", 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                weight='bold'
                )

    # Set the plot parameters
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_yticks(np.arange(0,max([m[2] for m in mr1s])+1.0,0.5))
    ax.set_ylabel("MR1@90 (↓)", fontsize=15)
    #ax.set_title(models[0][0].split("-")[-1].replace("_"," "), fontsize=17)
    ax.grid()
    ax.legend(loc=4, fontsize=10, title="Embedding, Search Combinations", title_fontsize=11, fancybox=True)
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        names = "-".join([model[0] for model in models])
        fig_path = os.path.join(save_dir, f"{names}-MR1_comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_map_comparisons_single_variation(models, eval_dir=EVAL_DIR, fig_name="", 
                                          save_fig=False, save_dir=FIGURES_DIR,):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "mAP.csv")
        df = pd.read_csv(map_path)
        maps.append((model[0], model[1], df.mAP.to_numpy()))
    K = df.k.to_numpy()

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Performances using mAP@15 values on {DATASET_NAME}", fontsize=20, weight='bold')
    #ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    ax.set_title("Page 1 Results", fontsize=15)
    #for i in range(len(K)):
    for j,(model_name,search,map) in enumerate(maps):
        ax.bar(0+positions[j], 
                map[0], 
                #label=model_name+f" {search}" if 0==0 else "", 
                label=get_model_name(model_name) if 0==0 else "",
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(0+positions[j], 
                map[0]+0.01, 
                f"{map[0]:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold'
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    #ax.set_xticks(np.arange(i+1), K)
    #ax.set_xticks(np.arange(1,i+2))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    #ax.set_xlabel("K (Similarity Rank)", fontsize=15)
    ax.set_xlabel("Embedding, Search Combinations", fontsize=15)
    #ax.set_xlabel("Page", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_ylim([0,1])
    #ax.set_title(models[0][0].split("-")[-1].replace("_"," "), fontsize=17)
    ax.grid()
    ax.legend(loc="best", fontsize=10, title_fontsize=11, fancybox=True)
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        if fig_name == "":
            names = "-".join([model[0] for model in models])
            fig_path = os.path.join(save_dir, f"{names}-mAP_comparison_k15.png")
        else:
            fig_path = os.path.join(save_dir, fig_name+".png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_map_comparisons(models, eval_dir=EVAL_DIR, 
                         save_fig=False, save_dir=FIGURES_DIR):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.25, 0.25, len(models))
    delta = positions[1]-positions[0]
    n_variations = len(models[1][1])

    fig,axs = plt.subplots(nrows=1, ncols=n_variations, 
                           figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Performances using mAP@k values Evaluated on {DATASET_NAME} Set", fontsize=20, weight='bold')
    for i in range(n_variations):

        # Read all the maps for all variations of model i
        maps = []
        for model in models:
            model_dir = os.path.join(eval_dir, f"{model[0]}-{model[1][i]}")
            results_dir = os.path.join(model_dir, model[2])
            map_path = os.path.join(results_dir, "mAP.csv")
            df = pd.read_csv(map_path)
            maps.append(df.mAP.to_numpy())
        K = df.k.to_numpy()

        # Plot the maps
        for j in range(len(K)):
            for l,map in enumerate(maps):
                if j==0:
                    label = models[l][0]
                else:
                    label = "" # Only dsiplay labels once
                axs[i].bar(j+positions[l], 
                        map[j], 
                        label=label, 
                        width=delta*0.85, 
                        color=COLORS[l], 
                        edgecolor='k'
                        )
                axs[i].text(j+positions[l], 
                            map[j]+0.01, 
                            f"{map[j]:.2f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=8, 
                            weight='bold')

        # Set the plot parameters
        axs[i].set_yticks(np.arange(0,1.05,0.05))
        axs[i].set_xticks(np.arange(j+1), K)
        axs[i].tick_params(axis='x', which='major', labelsize=13)
        axs[i].tick_params(axis='y', which='major', labelsize=11)
        if i==1:
            axs[i].set_xlabel("K (Similarity Rank)", fontsize=15)
        axs[i].set_ylabel("mAP@k (↑)", fontsize=15)
        axs[i].set_ylim([0,1])
        if i==n_variations-1:
            title="Original Size"
        else:
            title = " ".join(models[1][1][i].split("PCA_")[1].split("-")[0].split("_") + ["PCA Components"])
        axs[i].set_title(title, fontsize=17)
        axs[i].grid()
        axs[i].legend(fontsize=10, title="Models", title_fontsize=11, fancybox=True)

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        names = "-".join([model[0] for model in models])
        fig_path =os.path.join(save_dir, f"{names}-mAP_comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_av_label_based_map_comparisons(models, eval_dir=EVAL_DIR, fig_name="", 
                                        save_fig=False, save_dir=FIGURES_DIR):

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "av_label_based_mAP_at_15.txt")
        with open(map_path, "r") as f:
            map = float(f.read())
        maps.append((model[0], model[1], map))

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(f"Embedding Performances of Average Label-Based mAP@15 values on {DATASET_NAME}", fontsize=20, weight='bold')
    #ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    ax.set_title("Page 1 Results", fontsize=15)
    for j,(model_name,search,map) in enumerate(maps):
        ax.bar(0+positions[j], 
                map, 
                label=get_model_name(model_name) if 0==0 else "",
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(0+positions[j], 
                map+0.01, 
                f"{map:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold'
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("Embedding, Search Combinations", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15) # TODO: change name?
    ax.set_ylim([0,1])
    ax.grid()
    ax.legend(loc="best", fontsize=10, title_fontsize=11, fancybox=True)
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        if fig_name == "":
            names = "-".join([model[0] for model in models])
            fig_path = os.path.join(save_dir, f"{names}-av_label_based_mAP_comparison_k15.png")
        else:
            fig_path = os.path.join(save_dir, fig_name+".png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

#####################################################################################

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', 
                        type=str, 
                        required=True, 
                        help='Name of the model.')
    parser.add_argument("--save-dir",
                        type=str,
                        default=FIGURES_DIR,
                        help="Directory to save the figures.")
    args=parser.parse_args()

    plot_map_at_15(args.model, save_fig=True, save_dir=args.save_dir)
    plot_label_based_map(args.model, save_fig=True, save_dir=args.save_dir)
    plot_mr1(args.model, save_fig=True, save_dir=args.save_dir)
    print("Done!")
