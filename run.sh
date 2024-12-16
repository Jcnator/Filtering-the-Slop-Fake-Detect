#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <mode>"
    echo "Modes:"
    echo "  SINGLE_IMAGE - Process a single image"
    echo "  BATCH_VAL    - Perform batch validation"
    exit 1
fi

MODE=$1

case "$MODE" in
    SINGLE_IMAGE)
        echo "Processing a single image..."
        mkdir -p data/single_image_testing/

        cd data/single_image_testing/

        wget "https://i.ibb.co/z7bTQd4/will-smith-fake.jpg"

        wget "https://i.ibb.co/xLGB6fp/dude-real.png"

        cd ../

        cd ../

        echo "Running for Fake Image"

        python3 validate.py --arch=CLIP:ViT-L/14 --classifier=SVM --result_folder=results --image_path=data/single_image_testing/will-smith-fake.jpg

        echo "Running for Real Image"

        python3 validate.py --arch=CLIP:ViT-L/14 --classifier=SVM --result_folder=results --image_path=data/single_image_testing/dude-real.png
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


