"""Takes frame level yamnet embeddings and processes them for similarity
search. First aggregates frame level embeddings into clip level embeddings
then applies PCA to reduce the dimensions and finally normalizes by the
length."""

import os
import time
import glob
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from sklearn.decomposition import PCA

from directories import AUDIO_DIR

def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def aggregate_frames(embeds, aggregation="mean"):
    """ Takes a list of frame level embeddings and aggregates 
    them into a clip level embedding, if not already aggregated."""
    # Convert to numpy array
    if type(embeds)==list:
        embeds = np.array(embeds)
    # Aggreagate if multiple frames exist and specified
    if aggregation!="none" and len(embeds.shape)!=1:
        if aggregation=="mean":
            embeds = embeds.mean(axis=0)
        elif aggregation=="median":
            embeds = np.median(embeds, axis=0)
        elif aggregation=="max":
            embeds = embeds.max(axis=0)
    return embeds

def normalize_embedding(embeds):
    """Normalize the clip level embedding"""
    assert len(embeds.shape)==1, "Expects a 1D Clip Embedding"
    return embeds/np.linalg.norm(embeds)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                                   formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Directory containing embedding.json files.')
    parser.add_argument("-a", "-aggregation", 
                        type=str, 
                        choices=["mean", "median", "max", "none"], 
                        default="mean", 
                        help="Type of embedding aggregation.")
    parser.add_argument("-N", type=int, default=100, 
                        help="Number of PCA components to keep. -1 to do not apply.")
    parser.add_argument("--no-normalization",
                        action="store_true", 
                        help="Do not normalize clip embedding at the end.")
    parser.add_argument('--plot-scree', action='store_true', 
                        help="Plot variance contributions of PCA components.")
    args=parser.parse_args()

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.path, "*.json"))
    print(f"{len(embed_paths)} embeddings were found in the directory.")

    # Load the embeddings and process them
    print("Reading the embeddings and processing them...")
    start_time = time.time()
    embeddings = []
    for embed_path in embed_paths:
        with open(embed_path, 'r') as infile:
            model_outputs = json.load(infile)
        # Process and collect
        if model_outputs['embeddings'] is not None: # Filter out the None types
            clip_embedding = aggregate_frames(model_outputs["embeddings"], 
                                              aggregation=args.a)
            embeddings.append(clip_embedding)
    embeddings = np.array(embeddings)
    total_time = time.time()-start_time
    print(f"{len(embeddings)} embeddings were read.")
    print(f"Total pre-processing time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Create the output dir
    n_components = args.N if args.N!=-1 else embeddings.shape[1] # PCA components
    output_dir = f"{args.path}-Agg_{args.a}-PCA_{n_components}-Norm_{not args.no_normalization}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Scree plot
    # TODO: is this PCA effecting data?
    if args.plot_scree:
        print(f"Plotting the PCA Scree plot next to the embeddings...")
        import matplotlib.pyplot as plt
        model = os.path.basename(args.path)
        data = os.path.basename(os.path.dirname(args.path))
        title=f'FSD50K.{data} - {model} Embeddings PCA Scree Plot'
        pca = PCA(n_components=None, copy=True)
        pca.fit(embeddings)
        PC_values = np.arange(pca.n_components_) + 1
        cumsum_variance = 100*np.cumsum(pca.explained_variance_ratio_)
        fig,ax = plt.subplots(figsize=(15,8), constrained_layout=True)
        fig.suptitle(title, fontsize=20)
        ax.plot(PC_values, cumsum_variance, 'ro-', linewidth=2)
        ax.set_xlim([-5,len(PC_values)+5])
        ax.set_yticks(np.arange(0,105,5)) # 5% increase
        ax.set_xlabel('Number of Principal Components Selected', fontsize=15)
        ax.set_ylabel('% Cumulative Variance Explained', fontsize=15)
        ax.grid()
        figure_path = os.path.join(output_dir, f'FSD50K.{data}-{model}-scree_plot.jpeg')
        print(f"Exported the figure to: {figure_path}")
        fig.savefig(figure_path)

    # Apply PCA if specified
    if args.N!=-1:
        print("Applying PCA to each embedding...")
        start_time = time.time()
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        total_time = time.time()-start_time
        print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Normalize at the end if specified
    if not args.no_normalization:
        print("Normalizing embeddings...")
        start_time = time.time()
        embeddings = np.array([normalize_embedding(embed) for embed in embeddings])
        print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Export the transformed embeddings
    print("Exporting the embeddings...")
    for embed_path,embed in zip(embed_paths,embeddings):
        fname = get_file_name(embed_path)
        embed = {"audio_path": os.path.join(AUDIO_DIR,f"{fname}.wav"),
                "embeddings": embed.tolist()}
        output_path = os.path.join(output_dir, f"{fname}.json")
        with open(output_path, "w") as outfile:
            json.dump(embed, outfile, indent=4)

    #############
    print("Done!")