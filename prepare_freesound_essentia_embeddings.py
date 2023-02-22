import os
import argparse
import time
import glob
import yaml
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Use these statistics for each feature
PCA_DESCRIPTORS = [
    "mean",
    "dmean",
    "dmean2",
    "var",
    "dvar",
    "dvar2"
]

# Features that are multiple band
MBAND_FEATURES = [
    "barkbands",
    "erb_bands",
    "frequency_bands",
    "gfcc",
    "mfcc",
    "scvalleys",
    "spectral_contrast"
]

def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())

def select_subset(output):
    """ Selects a determined subset from a large set of features
    """
    # For features that have multiple bands, collect all statistics for each band separately
    mband_feats = {}
    for feat in MBAND_FEATURES:
        n_bands = len(output["lowlevel"][feat][PCA_DESCRIPTORS[0]]) # Get the Number of bands
        for i in range(n_bands): # Access each band
            mband_feats[f"{feat}_{i}"] = {}
            for stat in PCA_DESCRIPTORS:
                mband_feats[f"{feat}_{i}"][stat] = output["lowlevel"][feat][stat][i]
        del output["lowlevel"][feat]
    # Insert the collection to the rest
    for k,v in mband_feats.items():
        output["lowlevel"][k] = v

    # Select the subset of features
    embed = {}
    for feat,feat_dct in output["lowlevel"].items():
        if type(feat_dct) == dict:
            embed[feat] = []
            for stat in PCA_DESCRIPTORS:
                embed[feat].append(feat_dct[stat])

    return embed

AUDIO_DIR = "/data/FSD50K/FSD50K.eval_audio"

# TODO: remove AUDIO_PATH?
# TODO: whiten PCA??
if __name__=="__main__":

    parser=argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Directory containing fs-essentia-extractor_legacy embeddings.')
    parser.add_argument('--plot-scree', action='store_true', 
                        help="Plot the var contributions of PCA components.")
    parser.add_argument("-N", type=int, default=100, 
                        help="Number of PCA components to keep.")
    args = parser.parse_args()

    # Read all the embeddins
    embed_paths = glob.glob(os.path.join(args.path, "**", "*.yaml"), recursive=True)
    print(f"{len(embed_paths)} embeddings found.")

    # Create the initial embeddings from model outputs
    print("Creating the initial embeddings...")
    start_time = time.time()
    fnames,embeddings = [],[]
    for embed_path in embed_paths:
        # Get the fname from the path
        fnames += [get_file_name(embed_path).split("-")[0]]
        # Load the features and select the subset
        feat_dict = load_yaml(embed_path)
        embed = select_subset(feat_dict)
        embeddings += [embed]
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    SUBSET_KEYS = list(embeddings[0].keys()) # List of all included features
    print(f"{len(SUBSET_KEYS)} features selected.")

    # Create and store a Scaler for each feature
    print("Fitting scalers for each feature...")
    start_time = time.time()
    scalers = []
    for feat in SUBSET_KEYS:
        # Create the Data Matrix
        data = np.array([embed[feat] for embed in embeddings])
        scaler = MinMaxScaler()
        scaler.fit(data)
        scalers.append((feat,scaler))
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Normalize each feature independently
    print("Normalizing each feature independently...")
    start_time = time.time()
    for i in range(len(embeddings)):
        for key,scaler in scalers:
            d = np.array(embeddings[i][key]).reshape(1,-1)
            embeddings[i][key] = scaler.transform(d).reshape(-1)
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Concat all normalized features, make sure same order is followed
    print("Concatanating all the features....")
    start_time = time.time()
    embeddings = np.array([np.array([embed[k] for k in SUBSET_KEYS]).reshape(-1) for embed in embeddings])
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Apply PCA
    print("Applying PCA...")
    start_time = time.time()
    n_components = args.N
    if args.plot_scree: # For informative purposes keep principal components
        n_components= None
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Create the output dir
    model = os.path.basename(os.path.dirname(args.path))
    output_dir = args.path.replace(model,model+"_prepared")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}...")

    # Scree plot
    if args.plot_scree:
        print(f"Plotting PCA Scree plot to {output_dir}...")
        import matplotlib.pyplot as plt
        model = os.path.basename(os.path.dirname(args.path))
        data = os.path.basename(args.path)
        title=f'{model} - FSD50K.{data} Embeddings PCA Scree Plot'
        PC_values = np.arange(pca.n_components_) + 1
        fig,ax = plt.subplots(figsize=(15,8), constrained_layout=True)
        fig.suptitle(title, fontsize=20)
        ax.plot(PC_values, 100*np.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2)
        ax.set_xlim([-5,len(PC_values)+5])
        ax.set_xlabel('Number of Principal Components Selected', fontsize=15)
        ax.set_ylabel('% Cumulative Variance Explained', fontsize=15)
        ax.grid()
        figure_path = os.path.join(output_dir, f'{model}-FSD50K.{data}-scree_plot.jpeg')
        fig.savefig(figure_path)

    # Export the transformed embeddings
    for fname,embed in zip(fnames,embeddings):
        embed = {"audio_path": os.path.join(AUDIO_DIR,f"{fname}.wav"),
                "embeddings": embed.tolist()}
        output_path = os.path.join(output_dir, f"{fname}.json")
        with open(output_path, "w") as outfile:
            json.dump(embed, outfile, indent=4)