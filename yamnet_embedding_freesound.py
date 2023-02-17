import os
import glob
import json
import argparse

from essentia.standard import EasyLoader, TensorflowPredictVGGish

TRIM_DUR = 30
SAMPLE_RATE = 16000
ANALYZER_NAME = 'audioset-yamnet_v1'
MODEL_PATH = "models/yamnet/audioset-yamnet-1.pb"
AUDIO_EXT = ["ogg"] # TODO: wav?
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

def process_audio(model_embeddings, audio_path, output_dir=""):

    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, sampleRate=SAMPLE_RATE, endTime=TRIM_DUR, replayGain=0)
    audio = loader()

    # Process
    embedding = get_clip_embedding(model_embeddings, audio)

    # Save results
    if not output_dir: # If dir not specified
        export_path = f"{audio_path}.json" # next to the audio file
    else:
        sound_bank_dir = os.path.basename(os.path.dirname(os.path.dirname(audio_path))) # Bank of sounds
        query = os.path.basename(os.path.dirname(audio_path)) # The query name is the folder name
        export_dir = os.path.join(output_dir, sound_bank_dir, query)
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"{os.path.basename(audio_path)}.json")
    json.dump({
        'audio_path': audio_path,
        'embeddings': embedding
    }, open(export_path, 'w'), indent=4)

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='YAMNet Explorer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to an audio file or a directory containing audio files.')
    parser.add_argument('-o', '--output-dir', type=str, default=EMBEDDINGS_DIR, help="Save output files to a directory.")
    args=parser.parse_args()

    # Configure the embedding model
    model_embeddings = TensorflowPredictVGGish(graphFilename=MODEL_PATH, input="melspectrogram", output="embeddings")

    if os.path.isfile(args.path):
        process_audio(model_embeddings, args.path)
    else:
        # Search all the files and subdirectories for each AUDIO_EXT
        audio_paths = sum([glob.glob(args.path+f"/**/*.{ext}", recursive=True) for ext in AUDIO_EXT], [])
        for audio_path in audio_paths:
            process_audio(model_embeddings, audio_path, args.output_dir)

    #############
    print("Done!")