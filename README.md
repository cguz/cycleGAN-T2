# Description

Tensorflow 2 implementation of CycleGAN.

- This is the same implementation of Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf). Thus, all the credits to the Author: [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## Our modifications

- [x] Use Python environment
- [x] Sentinel dataset

## Usage

- Environment

  - Python 3.6

  - TensorFlow 2.2, TensorFlow Addons 0.10.0

  - OpenCV, scikit-image, tqdm, oyaml

  - sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip

    ```console
    python3 -m venv .env

    source .env/bin/activate
    ```

    Install requirements

    ```console
    pip install -r requirements.txt
    ```

- Dataset

  - download the opssat dataset

    ./download_dataset.sh datasets opssat 1

  - see [download_dataset.sh](./download_dataset.sh) for more datasets

- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sentinel
    ```

  - tensorboard for loss visualization

    ```console
    tensorboard --logdir ./output/sentinel/summaries --port 6006
    ```

- Example of testing

    ```console
    CUDA_VISIBLE_DEVICES=0 python test.py --experiment_dir ./output/sentinel
    ```
