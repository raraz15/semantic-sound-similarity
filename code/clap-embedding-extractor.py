"""Takes a FSD50K csv file specifying audio file names and computes embeddings 
using a CLAP https://github.com/LAION-AI/CLAP/tree/main .Commit ID: 6b1b4b5
All frame embeddings are exported as it is.

"""

import os
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

from lib.laion_clap import CLAP_Module
from lib.directories import AUDIO_DIR, GT_PATH, EMBEDDINGS_DIR

def process_audio(model_embeddings, audio_path, output_dir):
    """ Reads the audio of given path, creates the embeddings and exports.
    The model_embeddings should be a CLAP_Module instance. Different from TensorflowPredict
    models, CLAP models create a clip level embedding."""

    # Process
    embeddings = model_embeddings.get_audio_embedding_from_filelist(x=[audio_path], 
                                                                     use_tensor=False).tolist()
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path',
                        type=str, 
                        help="Path to model.pt chekpoint. Should point to models/")
    parser.add_argument('-o', 
                        '--output_dir', 
                        type=str, 
                        default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Configure the embedding mode w We append a "clap" to the start 
    # of the downloaded models
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    if model_name in ["clap-630k-audioset-fusion-best", "clap-630k-fusion-best"]:
        model = CLAP_Module(enable_fusion=True)
    elif "clap-music_speech_audioset_epoch_15_esc_89.98" == model_name:
        model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load the model
    model.load_ckpt(args.model_path)

    # Read the file names
    fnames = pd.read_csv(GT_PATH)["fname"].to_list()
    audio_paths = [os.path.join(AUDIO_DIR, f"{fname}.wav") for fname in fnames]
    print(f"There are {len(audio_paths)} audio files to process.")

    # Determine the output directory
    if args.output_dir=="":
        # If the default output directory is used add the AUDIO_DIR to the path
        output_dir = os.path.join(EMBEDDINGS_DIR, os.path.basename(AUDIO_DIR), model_name)
    else:
        output_dir = os.path.join(args.output_dir, model_name)
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        if i%1000==0:
            print(f"[{i:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")
        process_audio(model, audio_path, output_dir)
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!\n")