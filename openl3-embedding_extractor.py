"""Using an Open-L3 config and weights, extracts embeddings over FSD50K.eval_audio.
FSD50K.eval_audio CSV and audios paths are hard coded in directories.py. All frame 
embeddings are exported without aggregation.

Adapted from https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8
OpenL3 models are available at https://essentia.upf.edu/models/feature-extractors/openl3/
"""

import os
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

import essentia.standard as es
from essentia import Pool

from directories import AUDIO_DIR, GT_PATH, EMBEDDINGS_DIR

TRIM_DUR = 30 # seconds

class MelSpectrogramOpenL3:
    def __init__(self, hop_time, n_mels):
        self.hop_time = hop_time

        self.n_mels = n_mels
        self.sr = 48000
        self.frame_size = 2048
        self.hop_size = 242
        self.a_min = 1e-10
        self.d_range = 80
        self.db_ref = 1.0

        self.patch_samples = int(1 * self.sr)
        self.hop_samples = int(self.hop_time * self.sr)

        self.w = es.Windowing(
            size=self.frame_size,
            normalized=False,
        )
        self.s = es.Spectrum(size=self.frame_size)
        self.mb = es.MelBands(
            highFrequencyBound=self.sr / 2,
            inputSize=self.frame_size // 2 + 1,
            log=False,
            lowFrequencyBound=0,
            normalize="unit_tri",
            numberBands=self.n_mels,
            sampleRate=self.sr,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute(self, audio):

        batch = []
        for audio_chunk in es.FrameGenerator(
            audio, frameSize=self.patch_samples, hopSize=self.hop_samples
        ):
            melbands = np.array(
                [
                    self.mb(self.s(self.w(frame)))
                    for frame in es.FrameGenerator(
                        audio_chunk,
                        frameSize=self.frame_size,
                        hopSize=self.hop_size,
                        validFrameThresholdRatio=0.5,
                    )
                ]
            )

            melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
            melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
            melbands = np.maximum(melbands, melbands.max() - self.d_range)
            melbands -= np.max(melbands)

            batch.append(melbands.copy())
        return np.vstack(batch)

class EmbeddingsOpenL3:
    def __init__(self, graph_path, hop_time=1, batch_size=60, input_shape=[199,128]):
        self.hop_time = hop_time
        self.batch_size = batch_size

        self.graph_path = Path(graph_path)

        self.x_size = input_shape[0]
        self.y_size = input_shape[1]
        self.squeeze = False

        self.permutation = [0, 3, 2, 1]

        self.input_layer = "melspectrogram"
        self.output_layer = "embeddings"

        self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time,
                                                  n_mels=self.y_size,
                                                  )

        self.model = es.TensorflowPredict(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer],
            squeeze=self.squeeze,
        )

    def compute(self, audio):
        mel_spectrogram = self.mel_extractor.compute(audio)
        # in OpenL3 the hop size is computed in the feature extraction level

        hop_size_samples = self.x_size

        batch = self.__melspectrogram_to_batch(mel_spectrogram, hop_size_samples)

        pool = Pool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)
            pool.set(self.input_layer, batch[start:end])
            out_pool = self.model(pool)
            embeddings.append(out_pool[self.output_layer].squeeze())

        return np.vstack(embeddings)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        batch = es.TensorTranspose(permutation=self.permutation)(batch)
        return batch

# TODO: only discard non-floatable frames?
def create_embeddings(model, audio):
    """ Takes an embedding model and an audio array and returns the clip level embedding."""

    try:
        embeddings = model.compute(audio) # Embedding vectors of each frame
        embeddings = [[float(value) for value in embedding] for embedding in embeddings]
        return embeddings
    except AttributeError:
        return None

# TODO: L3 MonoLoader others
def process_audio(model_embeddings, audio_path, output_dir, sample_rate):
    """ Reads the audio of given path, creates the embeddings and exports."""

    # Load the audio file
    loader = es.EasyLoader()
    loader.configure(filename=audio_path, sampleRate=sample_rate, endTime=TRIM_DUR, replayGain=0)
    audio = loader()
    # Zero pad short clips
    if audio.shape[0] < sample_rate:
        audio = np.concatenate((audio, np.zeros((sample_rate-audio.shape[0]))))
    # Process
    embeddings = create_embeddings(model_embeddings, audio)
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)

if __name__ == "__main__":

    parser=ArgumentParser(description=__doc__, 
                                formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, 
                        help="Path to config.json file of the model. "
                        "Assumes the model.pb is next to it.")
    parser.add_argument('-o', '--output_dir', type=str, default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Read the config file
    with open(args.config, "r") as json_file:
        config = json.load(json_file)
    print("Config:")
    print(json.dumps(config, indent=4))

    # Configure the embedding model
    model_name = os.path.splitext(os.path.basename(args.config))[0]
    model_path = os.path.join(os.path.dirname(args.config), f"{model_name}.pb")
    model_embeddings = EmbeddingsOpenL3(model_path,
                                        input_shape=config['schema']['inputs'][0]['shape'])

    # Read the file names from AUDIO_DIR
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
        process_audio(model_embeddings, audio_path, output_dir, config['inference']['sample_rate'])
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!")