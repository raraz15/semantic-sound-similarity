"""Takes frame level model embeddings and processes them for similarity
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

from lib.utils import get_fname
from lib.directories import AUDIO_DIR

def aggregate_frames(embeds, aggregation="mean"):
    """ Takes a list of frame level embeddings and aggregates 
    them into a clip level embedding, if not already aggregated.
    Expects a numpy array of shape (n_frames, embedding_dim) or
    a list of numpy arrays of shape (embedding_dim,)."""

    # Convert to numpy array
    if type(embeds)==list:
        embeds = np.array(embeds)

    # Aggreagate if multiple frames exist and specified
    if aggregation!="none":
        if len(embeds.shape)!=1:
            if aggregation=="mean":
                embeds = embeds.mean(axis=0)
            elif aggregation=="median":
                embeds = np.median(embeds, axis=0)
            elif aggregation=="max":
                embeds = embeds.max(axis=0)
        else:
            print("Embeddings are already aggregated.")
    elif aggregation=="none":
        if len(embeds.shape)==1:
            pass
        elif embeds.shape[0]==1:
            # If only one frame exists, take that
            embeds = embeds[0]
        else:
            raise ValueError(f"Embeddings have wrong shape={embeds.shape}.")
    else:
        raise ValueError("Cannot aggregate the embeddings.")
    return embeds

def normalize_embedding(embedding):
    """Normalize the clip level embedding"""

    assert len(embedding.shape)==1, "Expects a 1D Clip Embedding"
    return embedding/np.linalg.norm(embedding)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('embed_dir', 
                        type=str, 
                        help='Directory containing embedding.json files.')
    parser.add_argument("-a", "-aggregation", 
                        type=str, 
                        choices=["mean", "median", "max", "none"], 
                        default="mean", 
                        help="Type of embedding aggregation.")
    parser.add_argument("-N", 
                        type=int, 
                        default=100, 
                        help="Number of PCA components to keep. -1 to do not apply.")
    parser.add_argument("--no-normalization",
                        action="store_true", 
                        help="Do not normalize the final clip embedding.")
    parser.add_argument("--normalization",
                        action="store_true", 
                        help="Normalize the final clip embedding.")
    parser.add_argument('--plot-scree', 
                        action='store_true', 
                        help="Plot variance contributions of PCA components.")
    parser.add_argument("--output-dir",
                        type=str,
                        default="",
                        help="Path to output directory. If not provided, "
                        "a directory will be created in the same directory "
                        "as the embed_dir.")
    args=parser.parse_args()

    if args.normalization and args.no_normalization:
        raise ValueError("Cannot specify both --normalization and --no-normalization")

    # Read all the json files in the tree
    embed_paths = glob.glob(os.path.join(args.embed_dir, "*.json"))
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
    assert len(embed_paths)==len(embeddings), \
        f"Number of embeddings and paths do not match. " \
        f"Embeddings: {len(embeddings)}, Paths: {len(embed_paths)}"
    embeddings = np.vstack(embeddings)
    total_time = time.time()-start_time
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Total pre-processing time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Determine PCA components
    n_components = args.N if args.N!=-1 else embeddings.shape[1] # PCA components

    # Determine the output directory and create it
    if args.output_dir=="":
        output_dir = f"{args.embed_dir}-Agg_{args.a}-PCA_{n_components}-Norm_{not args.no_normalization}"
    else:
        output_dir = os.path.join(args.output_dir, os.path.basename(args.embed_dir))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Prepared embeddings will be extracte to: {output_dir}")

    # Apply PCA if specified
    if args.N!=-1:
        print("Applying PCA to each embedding...")
        start_time = time.time()
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        assert embeddings.shape == (len(embed_paths), n_components), \
            f"PCA went wrong. Expected shape: {(len(embed_paths), n_components)}, " \
            f"Actual shape: {embeddings.shape}"
        total_time = time.time()-start_time
        print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")

    # Normalize at the end if specified
    if (not args.no_normalization) or args.normalization:
        print("Normalizing embeddings...")
        start_time = time.time()
        embeddings = np.vstack([normalize_embedding(embed) for embed in embeddings])
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Total time: {time.strftime('%M:%S', time.gmtime(total_time))}")
        # Control the normalization
        assert np.allclose(1, np.linalg.norm(embeddings, axis=1)), \
            "Embeddings are not normalized."

    # Export the transformed embeddings
    print("Exporting the embeddings...")
    for embed_path,embed in zip(embed_paths,embeddings):
        fname = get_fname(embed_path)
        embed = {"audio_path": os.path.join(AUDIO_DIR,f"{fname}.wav"),
                "embeddings": embed.tolist()}
        output_path = os.path.join(output_dir, f"{fname}.json")
        with open(output_path, "w") as outfile:
            json.dump(embed, outfile, indent=4)

    #############
    print("Done!\n")