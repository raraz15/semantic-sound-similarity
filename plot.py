import os
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from directories import FIGURES_DIR, EVAL_DIR
DATASET_NAME = "FSD50K.eval_audio"
EVAL_DIR = os.path.join(EVAL_DIR, DATASET_NAME)

colors = ["g", "b", "r", "y", "c", "m", "k"]

def plot_map(model, eval_dir=EVAL_DIR, n_cols=3, save_fig=False, save_dir=FIGURES_DIR):
    """Takes a model name and plots the MAP@K for all the variations of the model."""

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
    n_rows = len(k_values)//n_cols

    # Plot the maps
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, 
                            figsize=(6*n_cols,6*n_rows), constrained_layout=True)
    fig.suptitle(model, fontsize=20, weight='bold')
    for n,k in enumerate(k_values):

        row, col = n//n_cols, n%n_cols
        xticks = []
        for j,search in enumerate(map_dict[k].keys()):
            for z,(variation,map) in enumerate(map_dict[k][search]):
                if j==0:
                    xticks.append(variation)
                if z==0:
                    if search=="dot":
                        label = "dot product"
                    elif search=="nn":
                        label = "nearest neighbors"
                else:
                    label = ""
                axs[row,col].bar(z+positions[j], 
                                height=map, 
                                width=delta*0.8, 
                                label=label, 
                                color=colors[j], 
                                edgecolor='k')
                axs[row,col].text(z+positions[j], 
                                map+0.01, 
                                f"{map:.3f}", 
                                ha='center', 
                                va='bottom', 
                                fontsize=10)

        axs[row,col].set_title(f"k={k}", fontsize=17, weight='bold')
        axs[row,col].tick_params(axis='y', which='major', labelsize=11)
        axs[row,col].tick_params(axis='x', which='major', labelsize=10)
        axs[row,col].set_xticks(np.arange(len(xticks)), xticks, rotation=20)
        axs[row,col].set_yticks(np.arange(0,1.05,0.05))
        axs[row,col].grid()
        axs[row,col].set_ylabel("MAP@K", fontsize=15)
        axs[row,col].set_xlabel("Processing Parameters", fontsize=15)
        axs[row,col].legend(fontsize=11, loc=4, title="Search Algorithms", 
                            title_fontsize=12, fancybox=True)
        axs[row,col].set_ylim([0,1])
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{model}-mAP.png"))
    plt.show()

def plot_mr1(model, eval_dir=EVAL_DIR, save_fig=False, save_dir=FIGURES_DIR):
    """Takes a model name and plots the MR1 for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))

    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

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
    fig.suptitle(model, fontsize=20, weight='bold')
    ax.set_title("MR1 Values of Embedding Aggregations and Search Algorithms", fontsize=17)
    xticks, max_val = [], []
    for i in range(len(variation_paths)):
        for j,search in enumerate(searches):
            variation, mr1 = mr1_dict[search][i]
            max_val += [mr1]
            if j%len(searches)==0:
                xticks.append(variation)
            if i==0:
                if search=="dot":
                    label = "dot product"
                elif search=="nn":
                    label = "nearest neighbors"
            else:
                label = ""
            ax.bar(i+positions[j], 
                height=mr1, 
                width=0.35, 
                label=label, 
                color=colors[j], 
                edgecolor='k')
            ax.text(i+positions[j], 
                    mr1+0.01, 
                    f"{mr1:.2f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10)

    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,max(max_val)+0.5,0.5))
    ax.set_ylabel("MR1@90", fontsize=15) # TODO: read K
    ax.set_xlabel("Processing Parameters", fontsize=15)
    ax.legend(loc=1, fontsize=11, title="Search Algorithms", 
              title_fontsize=12, fancybox=True)
    ax.grid()
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{model}-MR1.png"))
    plt.show()

def plot_map_comparisons_single_variation(models, eval_dir=EVAL_DIR, save_fig=False, save_dir=FIGURES_DIR):
    """Takes a list of models and plots the MAP@K for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.25, 0.25, len(models))
    delta = positions[1]-positions[0]

    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "mAP.csv")
        df = pd.read_csv(map_path)
        maps.append((model[0], df.mAP.to_numpy()))
    K = df.k.to_numpy()

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle("Model Comparison", fontsize=20, weight='bold')

    for i in range(len(K)):
        for j,(model_name,map) in enumerate(maps):
            ax.bar(i+positions[j], 
                    map[i], 
                    label=model_name if i==0 else "", 
                    width=delta*0.85, 
                    color=colors[j], 
                    edgecolor='k'
                    )
            ax.text(i+positions[j], 
                    map[i]+0.01, 
                    f"{map[i]:.2f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    weight='bold'
                    )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.set_xticks(np.arange(i+1), K)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("K", fontsize=15)
    ax.set_ylabel("MAP@K", fontsize=15)
    ax.set_ylim([0,1])
    ax.set_title(models[0][0].split("-")[-1].replace("_"," "), fontsize=17)
    ax.grid()
    ax.legend(fontsize=10, title="Models", title_fontsize=11, fancybox=True)
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        names = "-".join([model[0] for model in models])
        fig.savefig(os.path.join(save_dir, f"{names}-mAP_comparison_k15.png"))
    plt.show()

def plot_map_comparisons(models, eval_dir=EVAL_DIR, save_fig=False, save_dir=FIGURES_DIR):
    """Takes a list of models and plots the MAP@K for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.25, 0.25, len(models))
    delta = positions[1]-positions[0]
    n_variations = len(models)

    fig,axs = plt.subplots(nrows=1, ncols=n_variations, 
                           figsize=(18,6), constrained_layout=True)
    fig.suptitle("Model Comparison", fontsize=20, weight='bold')
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
                        color=colors[l], 
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
        axs[i].set_xlabel("K", fontsize=15)
        axs[i].set_ylabel("MAP@K", fontsize=15)
        axs[i].set_ylim([0,1])
        if i==n_variations-1:
            title="No PCA"
        else:
            title = " ".join(models[1][1][i].split("PCA_")[1].split("-")[0].split("_") + ["Components"])
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

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', 
                        type=str, 
                        required=True, 
                        help='Name of the model.')
    parser.add_argument("--compare",
                        type=list,
                        default=None,)
    args=parser.parse_args()

    plot_map(args.model, save_fig=True)
    plot_mr1(args.model, save_fig=True)
    print("Done!")
