"""Takes a FSD50K csv file specifying audio file names and computes embeddings 
using audioset-yamnet_v1. All frame embeddings are exported without aggregation."""

import os
import time
import json
import argparse

import numpy as np
import pandas as pd

from essentia.standard import EasyLoader, TensorflowPredictVGGish

from directories import AUDIO_DIR, GT_PATH

TRIM_DUR = 30
SAMPLE_RATE = 16000
ANALYZER_NAME = 'audioset-yamnet_v1'
MODEL_PATH = "models/audioset-yamnet/audioset-yamnet-1.pb"
EMBEDDINGS_DIR = f"data/embeddings/eval/{ANALYZER_NAME}"

# TODO: only discard non-floatable frames?
def create_embeddings(model, audio):
    """ Takes an embedding model and an audio array and returns the clip level embedding.
    """
    try:
        embeddings = model(audio) # Embedding vectors of each frame
        embeddings = [[float(value) for value in embedding] for embedding in embeddings]
        return embeddings
    except AttributeError:
        return None

# TODO: effect of zero padding short clips?
# TODO: energy based frame filtering (at audio input)
def process_audio(model_embeddings, audio_path, output_dir):
    """ Reads the audio of given path, creates the embeddings and exports."""
    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, sampleRate=SAMPLE_RATE, endTime=TRIM_DUR, replayGain=0)
    audio = loader()
    # Zero pad short clips
    if audio.shape[0] < SAMPLE_RATE:
        audio = np.concatenate((audio, np.zeros((SAMPLE_RATE-audio.shape[0]))))
    # Process
    embeddings = create_embeddings(model_embeddings, audio)
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)

if __name__=="__main__":

    parser=argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, default=GT_PATH, 
                        help='Path to csv file containing fnames.')
    args=parser.parse_args()

    # Configure the embedding model
    model_embeddings = TensorflowPredictVGGish(graphFilename=MODEL_PATH, 
                                               input="melspectrogram", 
                                               output="embeddings")

    # Read the file names
    fnames = pd.read_csv(args.path)["fname"].to_list()
    audio_paths = [os.path.join(AUDIO_DIR, f"{fname}.wav") for fname in fnames]
    print(f"There are {len(audio_paths)} audio files to process.")

    # Create the output directory
    output_dir = EMBEDDINGS_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        if i%1000==0:
            print(f"[{i:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")
        process_audio(model_embeddings, audio_path, output_dir)
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!")