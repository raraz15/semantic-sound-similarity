#!/bin/bash

MODELS_DIR="/home/roguz/freesound/freesound-perceptual_similarity/models/"
mkdir -p $MODELS_DIR

############################################################################################################

echo "====================================================================================================="
echo "Downloading VGGish model"
YAMNET_DIR="${MODELS_DIR}yamnet/"
echo $YAMNET_DIR
mkdir -p $YAMNET_DIR

urls=(
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P $YAMNET_DIR
done

############################################################################################################

echo
echo "====================================================================================================="
echo "Downloading FSD SID Net Model"
FSD_SINET_DIR="${MODELS_DIR}fsd-sinet/"
echo $FSD_SINET_DIR
mkdir -p $FSD_SINET_DIR

urls=(
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.json"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P $FSD_SINET_DIR
done
echo "====================================================================================================="
echo "Done!"

############################################################################################################