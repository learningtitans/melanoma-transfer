#!/bin/bash

if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$2" == "" ];  then
	echo
	echo "convertEDRAAtlas  <EDRAS CD-ROM Image path>  <target path>"
	echo
	echo "    <target path> will be created, the images will be copied"
	echo "    from <EDRAS CD-ROM path>, converted and cropped."
	echo
	echo "    <EDRAS CD-ROM path> should point to the Images folder of the"
	echo "    Interactive Atlas of Dermoscopy/ EDRA (ISBN: 88-86457-30-8),"
	echo "    usually ./Images (e.g. /Volumes/IAD/Images or /mnt/cdrom/Images"
	echo "    or /media/cdrom/Images etc.)"
	echo
	exit 1
else
	rootFolder=$1
fi 

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

