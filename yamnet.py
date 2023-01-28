import os
import glob
import json
import argparse

from essentia.standard import EasyLoader, TensorflowPredictVGGish

TRIM_DUR = 30
SAMPLE_RATE = 16000
ANALYZER_NAME = 'audioset-yamnet_v1'
MODEL_PATH = "models/yamnet/audioset-yamnet-1.pb"
CLASSES_PATH = "models/yamnet/audioset-yamnet-1.json"
AUDIO_EXT = ["ogg"] # TODO: wav?
EMBEDDINGS_DIR = "embeddings"

def get_classes(model, audio, class_names):
    """ Extracts class activations and generates the class vector
    """
    try:
        activations = model(audio).mean(axis=0)
        class_probabilities = [(class_name, float(activations[i])) for i, class_name in enumerate(class_names)]
        sorted_class_probabilities = sorted(class_probabilities, key=lambda x: x[1], reverse=True)
        max_probability = sorted_class_probabilities[0][1]
        classes = []
        if max_probability > 0.15:
            # Only output classes if the class with maximum probability is above a global threshold
            # For the rest, only add class names of those whose probability is at 0.95 or above than the highest probability
            probability_threshold = max_probability * 0.95
            classes = [class_name for class_name, probability in sorted_class_probabilities if probability >= probability_threshold]
    except AttributeError:
        classes = None
        sorted_class_probabilities = None
    return classes, sorted_class_probabilities

def get_embeddings(model, audio):
    """ Takes an embedding model and an audio file and returns the frame averaged embedding.
    """
    # Extract embeddings (TODO: can we do activations and embeddings together?)
    try:
        embeddings = model(audio).mean(axis=0)  # Take mean of 1-second frame embeddings
        embeddings = [float(value) for value in embeddings]  # Needs to be a list of non-np types so that JSON can encode it
    except AttributeError:
        embeddings = None
    return embeddings

def process_audio(model_activations, model_embeddings, class_names, audio_path, output_dir=""):

    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, sampleRate=SAMPLE_RATE, endTime=TRIM_DUR, replayGain=0)
    audio = loader()

    # Process
    classes, sorted_class_probabilities = get_classes(model_activations, audio, class_names)
    embeddings = get_embeddings(model_embeddings, audio)

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
        'classes': classes,
        'top_10_classes_probabilities': sorted_class_probabilities[:10] if sorted_class_probabilities is not None else None,
        'embeddings': embeddings
    }, open(export_path, 'w'), indent=4)

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='YAMNet Explorer.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to an audio file or a directory containing audio files.')
    parser.add_argument('-o', '--output-dir', type=str, default=EMBEDDINGS_DIR, help="Save output files to a directory.")
    args=parser.parse_args()

    # Configure the activation and the embedding models
    model_activations = TensorflowPredictVGGish(graphFilename=MODEL_PATH, input="melspectrogram", output="activations")
    model_embeddings = TensorflowPredictVGGish(graphFilename=MODEL_PATH, input="melspectrogram", output="embeddings")
    class_names = json.load(open(CLASSES_PATH))['classes']

    if os.path.isfile(args.path):
        process_audio(model_activations, model_embeddings, class_names, args.path)
    else:
        # Search all the files and subdirectories for each AUDIO_EXT
        audio_paths = sum([glob.glob(args.path+f"/**/*.{ext}", recursive=True) for ext in AUDIO_EXT], [])
        for audio_path in audio_paths:
            process_audio(model_activations, model_embeddings, class_names, audio_path, args.output_dir)

    #############
    print("Done!")