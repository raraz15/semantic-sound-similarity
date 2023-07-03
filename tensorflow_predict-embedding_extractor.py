"""Takes a FSD50K csv file specifying audio file names and computes embeddings 
using FSD-Sinet. All frame embeddings are exported without aggregation."""

import os
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd

from essentia.standard import EasyLoader, TensorflowPredictFSDSINet, TensorflowPredictVGGish

from directories import AUDIO_DIR, GT_PATH, EMBEDDINGS_DIR

TRIM_DUR = 30 # seconds

def create_embeddings(model, audio):
    """ Takes an embedding model and an audio array and returns the frame level embeddings.
    If the model produces a non-floatable embedding, returns None. This does not happen
    with models such as FSD-Sinet or VGGish, YamNet, OpenL3 on FSD50K eval."""
    try:
        embeddings = model(audio) # Embedding vectors of each frame
        embeddings = [[float(value) for value in embedding] for embedding in embeddings]
        return embeddings
    except AttributeError:
        print("Model produced a non-floatable embedding.")
        return None

# TODO: effect of zero padding short clips?
# TODO: effect of normalization?
def process_audio(model_embeddings, audio_path, output_dir, sample_rate):
    """ Reads the audio of given path, creates the embeddings and exports."""
    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, 
                     sampleRate=sample_rate, 
                     endTime=TRIM_DUR, # FSD50K are already below 30 seconds
                     replayGain=0 # Do not normalize the audio
                     )
    audio = loader()
    # Zero pad short clips (IN FSD50K 7% of the clips are shorter than 1 second)
    if audio.shape[0] < sample_rate:
        audio = np.concatenate((audio, np.zeros((sample_rate-audio.shape[0]))))
    # Process
    embeddings = create_embeddings(model_embeddings, audio)
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_path',
                        type=str, 
                        help="Path to config.json file of the model. "
                        "Assumes the model.pb is next to it.")
    parser.add_argument('-o', 
                        '--output_dir', 
                        type=str, 
                        default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Read the config file
    with open(args.config_path, "r") as json_file:
        config = json.load(json_file)
    print("Config:")
    print(json.dumps(config, indent=4))

    # Configure the embedding model
    model_name = os.path.splitext(os.path.basename(args.config_path))[0]
    model_path = os.path.join(os.path.dirname(args.config_path), f"{model_name}.pb")
    if "audioset-yamnet" in 'model_name':
        model_embeddings = TensorflowPredictVGGish(graphFilename=model_path, 
                                                input="melspectrogram", 
                                                output="embeddings")
    elif "audioset-vggish" in model_name:
        model_embeddings = TensorflowPredictVGGish(graphFilename=model_path, 
                                                output="model/vggish/embeddings")
    elif "fsd-sinet" in model_name:
        model_embeddings = TensorflowPredictFSDSINet(graphFilename=model_path,
                                                    output="model/global_max_pooling1d/Max")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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
        process_audio(model_embeddings, audio_path, output_dir, config['sample_rate'])
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!")