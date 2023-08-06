MAIN_PATH=$1
FILE=$2
URL_ID=$3
mkdir -p $MAIN_PATH

if [[ $FILE != "opssat" && $FILE != "sentinel" && $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: sentinel, apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

if [[ $FILE == "opssat" ]]; then

    URL='1aY_u6Y_VrAy4qagUZBpOXmo95_-BTuto'

    ZIP_FILE=./$MAIN_PATH/$FILE.tar.gz
    gdown --id $URL -O $ZIP_FILE

    TARGET_DIR=./$MAIN_PATH/$FILE/
    mkdir -p $TARGET_DIR
    tar -zxf $ZIP_FILE --directory ./$TARGET_DIR/
    mv ./$TARGET_DIR/denoiser_training_data/* ./$TARGET_DIR/
    rm -rf ./$TARGET_DIR/denoiser_training_data/
    rm $ZIP_FILE

else
    if [[ $URL_ID == "1" ]]; then
        URL='16cOelqwMmMo3Z3S9cb7_RMcMNWFGn4xx'
    fi
    if [[ $URL_ID == "1-output" ]]; then
        URL='1PxkLUTOBHczZyevy7MfhB5eQccKI16lg'
    fi

    ZIP_FILE=./$MAIN_PATH/$FILE.zip
    if [[ $FILE == "sentinel" ]]; then
        gdown --id $URL -O $ZIP_FILE
    else
        URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
        wget -N $URL -O $ZIP_FILE
    fi
    TARGET_DIR=./$MAIN_PATH/$FILE/
    mkdir $TARGET_DIR
    unzip $ZIP_FILE -d ./$MAIN_PATH/
    rm $ZIP_FILE
fi
