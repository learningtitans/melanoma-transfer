#!/usr/bin/env bash

# Exit on first error
set -e

function convertEDRAAtlas() {
    pushd `pwd` > /dev/null

    # Creates target folder
    mkdir "$2"
    mkdir "$2"/originals
    mkdir "$2"/ready
    
    # Copies all images into a "flat" folder
    cd "$2"/originals
    find "$1" -iname '*.jpg' -exec cp '{}' . \;
    
    # Renames all images to lowercase
    for SRC in `find . -iname '*.jpg'`; do
        DST=`dirname "${SRC}"`/`basename "${SRC}" | tr '[A-Z]' '[a-z]'`
        if [ "${SRC}" != "${DST}" ]
        then
            mv "${SRC}" "${DST}"
        fi
    done
    
    # Removes black borders from images
    find . -iname '*.jpg' -exec mogrify -path ../ready/ -bordercolor black -border 1x1 -fuzz 20% -trim +repage -gravity Center -crop 95%x95%+0+0 +repage '{}'  \;
    
    popd  > /dev/null
    
    echo
    echo The dataset path is: $2/ready
    echo
}

function fetch_file() {
    filename="$1"
    url="$2"
    if [ -e "$filename" ]; then
        echo "$url: file already downloaded (remove $filename to force re-download)"
    else
        echo "$url: fetching..."
        wget -O "$filename" "$url"
        echo "$url: done."
    fi
}

if [ "$1" = "--imagenet" ]; then
    mkdir -p datasets/imagenet/
    fetch_file datasets/imagenet/imagenet-vgg-m.mat http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat
    echo "converting from .mat file to .pkl..."
    python src/utils/mat2py.py
elif ["$1" = "--retina"]; then
    mkdir -p datasets/retina/
    python src/utils/convert.py --directory "$3" --convert_directory "$5"
elif ["$1" = "--melanoma"]; then
    mkdir -p "$5"
    python src/utils/resizing.py --input_dir "$3" --output_dir "$5"
else
    echo
    echo "USAGE: ./fetch.sh [OPTION] --infolder [INPUT_FOLDER] --outfolder [OUTPUT_FOLDER]"
    echo "WHERE: "
    echo "  [OPTION]  : '--imagenet' or '--retina' or '--melanoma' "
    echo "  [INPUT_FOLDER]    : path to raw retina/melanoma images "
    echo "  [OUTPUT_FOLDER]    : path to resized retina/melanoma images "
    echo ""
    echo "  ATTENTION! This script assumes that the dependencies are settled in the path."
    echo "  -> See README file for the list of dependencies;"
fi
