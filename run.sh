#!/bin/bash

TERMINAL_WIDTH=$(tput cols)

if [ -z "$1" ]; then
    echo "Usage: $0 <mode> [arch] [classifier]"
    echo "Modes:"
    echo "  SINGLE_IMAGE - Process a single image"
    echo "  BATCH_VAL    - Perform batch validation"
    echo "Optional:"
    echo "  arch         - Model architecture (default: CLIP:ViT-L/14)"
    echo "  classifier   - Classifier type (SVM or Linear, default: SVM)"
    exit 1
fi

MODE=$1
ARCH=${2:-"CLIP:ViT-L/14"}
CLASSIFIER=${3:-"SVM"}

case "$MODE" in
    SINGLE_IMAGE)
        echo "Processing a single image..."
        mkdir -p data/single_image_testing/

        wget -P data/single_image_testing/ "https://i.ibb.co/z7bTQd4/will-smith-fake.jpg"

        wget -P data/single_image_testing/ "https://i.ibb.co/xLGB6fp/dude-real.png"

        printf '%*s\n' "$TERMINAL_WIDTH" '' | tr ' ' '-'

        echo "RUNNING FOR FAKE IMAGE"

        python3 validate.py --arch="$ARCH" --classifier="$CLASSIFIER" --result_folder=results --image_path=data/single_image_testing/will-smith-fake.jpg

        printf '%*s\n' "$TERMINAL_WIDTH" '' | tr ' ' '-'
        echo "RUNNING FOR REAL IMAGE"

        python3 validate.py --arch="$ARCH" --classifier="$CLASSIFIER" --result_folder=results --image_path=data/single_image_testing/dude-real.png
        ;;
    BATCH_VAL)
        echo "TBD, as we'll need to pull large data and do some moving around."
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Valid modes are SINGLE_IMAGE or BATCH_VAL"
        exit 1
        ;;
esac


