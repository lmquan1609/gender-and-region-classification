# Config
Need to install some pip
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# if linux is installed; otherwise; don't need to install
pip install pickle5

pip install tensorboardx
```

# Data Preparation
1. Download the [data](https://drive.google.com/file/d/1hKiioqSPhk4n5Xz-KbNBrZXBWlBFOABP/view?usp=sharing) and then unzip it in the root directory of repository

2. Convert data to images of mel-spectrogram which is stored in pickle files

```
python preprocessing.py --data-dir data/wav16000/ --store-dir data/3seconds/store_spectrograms/
```

# Training
```
python train.py --config-path config/densenet_config.json
```

# Inference
To generate the accuracy on testing dataset
```
python inference.py --config-path config/densenet_config.json --model-path path/to/pytorch/model
```