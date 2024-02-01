"""Takes fs-essentia-extractor_legacy embeddings, and to implements
the Gaia feature preprocessing with Python. Namely, select a subset
 of the features, normalizes each of them independently and applies
 dimensionality reduction with PCA."""

import os
import time
import glob
import yaml
import json
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from lib.utils import get_fname
from lib.directories import AUDIO_DIR

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

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())

def select_subset(output):
    """ Selects a determined subset from a large set of features"""

    # For multiband features, collect PCA_DESCRIPTORS statistics of each band separately
    mband_feats = {}
    for feat in MBAND_FEATURES:
        n_bands = len(output["lowlevel"][feat][PCA_DESCRIPTORS[0]]) # Get the Number of bands
        for i in range(n_bands): # Access each band
            mband_feats[f"{feat}_{i}"] = {}
            for stat in PCA_DESCRIPTORS:
                mband_feats[f"{feat}_{i}"][stat] = output["lowlevel"][feat][stat][i]
        del output["lowlevel"][feat]
    # Insert the collection to the rest of the lowlevel features
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

# TODO: whiten PCA??
if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('embed_dir',
                        type=str,
                        help='Directory containing fs-essentia-extractor_legacy embeddings.')
    parser.add_argument("-N",
                        type=int,
                        default=100, 
                        help="Number of PCA components to keep. Pass -1 to skip PCA reduction.")
    parser.add_argument("--output-dir",
                        type=str,
                        default="",
                        help="Path to output directory. If not provided, "
                        "a directory will be created in the same directory "
                        "as the embed_dir.")
    args = parser.parse_args()

    # Read all the embeddins
    embed_paths = glob.glob(os.path.join(args.embed_dir, "*.yaml"))
    print(f"{len(embed_paths)} embeddings found.")

    # Create the initial embeddings from model outputs
    print("Selecting the features and concatenating...")
    start_time = time.time()
    fnames,embeddings = [],[]
    for i,embed_path in enumerate(embed_paths):
        # Get the fname from the path
        fnames += [get_fname(embed_path).split("-")[0]]
        # Load the features and select the subset
        feat_dict = load_yaml(embed_path)
        # Hand-pick the features
        embed = select_subset(feat_dict)
        # Use the first item to decide the order of concatenation
        if i==0:
            SUBSET_KEYS = list(embed.keys()) # List of all included features
            print(f"{len(SUBSET_KEYS)} features selected.")
        embed = np.array([embed[k] for k in SUBSET_KEYS]).reshape(-1)
        # Append the concatenated array
        embeddings += [embed]
        if (i+1)%1000==0 or i==0 or i+1==len(embed_paths):
            print(f"Processed [{i+1}/{len(embed_paths)}] embeddings...")
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Normalize each feature independently. It's actually MinMax scaling
    print("Normalizing the features...")
    start_time = time.time()
    scaler = MinMaxScaler()
    embeddings = scaler.fit_transform(embeddings)
    total_time = time.time()-start_time
    print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Control the normalization
    _min = embeddings.min(axis=0)
    _max = embeddings.max(axis=0)
    assert np.allclose(_min, 0) and np.allclose(_max, 1), f"Min-max scaling went wrong.\nmin={_min}, max={_max}"

    # Determine PCA components
    n_components = args.N if args.N!=-1 else embeddings.shape[1]
    # Apply PCA if specified
    if args.N!=-1:
        print("Applying PCA to each embedding...")
        start_time = time.time()
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        total_time = time.time()-start_time
        print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Create the output dir
    if args.output_dir == "":
        output_dir = f"{args.embed_dir}-PCA_{n_components}"
    else:
        output_dir = os.path.join(args.output_dir, 
                                  f"{os.path.basename(args.embed_dir)}-PCA_{n_components}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Embeddings will be extracted to: {output_dir}")

    # Export the transformed embeddings
    print("Exporting the embeddings...")
    for fname,embed in zip(fnames,embeddings):
        embed = {"audio_path": os.path.join(AUDIO_DIR,f"{fname}.wav"),
                "embeddings": embed.tolist()}
        output_path = os.path.join(output_dir, f"{fname}.json")
        with open(output_path, "w") as outfile:
            json.dump(embed, outfile, indent=4)

    #############
    print("Done!\n")