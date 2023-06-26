#!/bin/bash

MODELS_DIR="/home/roguz/freesound/freesound-perceptual_similarity/models/"
mkdir -p $MODELS_DIR

############################################################################################################

echo "====================================================================================================="
echo "Downloading OpenL3 model"

urls=(
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.json"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.json"\

)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P $MODELS_DIR
done

############################################################################################################

echo "====================================================================================================="
echo "Downloading YAMNet model"

urls=(
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb"\
    "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P $MODELS_DIR
done

############################################################################################################

echo "====================================================================================================="
echo "Downloading VGGish model"

urls=(
    "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb"\
    "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.json"\
)

for url in ${urls[@]}; do
    echo "$url"
    wget $url -P $MODELS_DIR
done

############################################################################################################


echo
echo "====================================================================================================="
echo "Downloading FSD SID Net Model"

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
    wget $url -P $MODELS_DIR
done
echo "====================================================================================================="
echo "Done!"

############################################################################################################