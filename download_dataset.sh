mkdir datasets
FILE=$1

if [[ $FILE != "sentinel" && $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: sentinel, apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

ZIP_FILE=./datasets/$FILE.zip
if [[ $FILE == "sentinel" ]]; then
    pip install gdown
    URL='16cOelqwMmMo3Z3S9cb7_RMcMNWFGn4xx'
    gdown --id $URL -O datasets/
else
    URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
    wget -N $URL -O $ZIP_FILE
fi
TARGET_DIR=./datasets/$FILE/
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
