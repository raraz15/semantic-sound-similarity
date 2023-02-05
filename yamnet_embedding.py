import os
import time
import json
import argparse

import pandas as pd

from essentia.standard import EasyLoader, TensorflowPredictVGGish

TRIM_DUR = 30
SAMPLE_RATE = 16000
ANALYZER_NAME = 'audioset-yamnet_v1'
MODEL_PATH = "models/yamnet/audioset-yamnet-1.pb"
EMBEDDINGS_DIR = f"embeddings/{ANALYZER_NAME}"

# TODO: frame aggregation, frame filtering, PCA
# TODO: only discard non-floatable frames?
def get_clip_embedding(model, audio):
    """ Takes an embedding model and an audio array and returns the clip level embedding.
    """
    try:
        embedding = model(audio).mean(axis=0)  # Take mean of 1-second frame embeddings
        embedding = [float(value) for value in embedding] # Needs to be a list of non-np types so that JSON can encode it
    except AttributeError:
        embedding = None
    return embedding

def process_audio(model_embeddings, audio_path, output_dir):

    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, sampleRate=SAMPLE_RATE, endTime=TRIM_DUR, replayGain=0)
    audio = loader()

    # Process
    embedding = get_clip_embedding(model_embeddings, audio)

    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 'embeddings': embedding}, outfile, indent=4)

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='YAMNet Explorer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to csv file containing audio paths.')
    args=parser.parse_args()

    # Configure the embedding model
    model_embeddings = TensorflowPredictVGGish(graphFilename=MODEL_PATH, input="melspectrogram", output="embeddings")

    # Read the labels and file names
    audio_paths = pd.read_csv(args.path)["path"].to_list()
    print(f"There are {len(audio_paths)} files to process.")

    # Create the output directory
    subset = os.path.splitext(os.path.basename(args.path))[0]
    output_dir = os.path.join(EMBEDDINGS_DIR, subset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        print(f"[{i}/{len(audio_paths)}]")
        process_audio(model_embeddings, audio_path, output_dir)
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!")