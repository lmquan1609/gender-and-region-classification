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

2. Convert data to low level feature which is stored in xlsx file and stored raw data in pickle file

2.a Stored raw data in pickle file
```
python preprocessdata/datatransformWaveMsNet.py --audiofolder "./3seconds" --samplingrate 16000 --traintxt "./pathsofaudio/train.txt" --validtxt "./pathsofaudio/valid.txt" --testtxt "./pathsofaudio/test.txt" --folderpickle "./Final/wavemsdata"
```
2.b Extract low level feature and stored feature in xlsx file
```
python preprocessdata/datatransformllNet.py --f "llfeature" --dstfoldername "./Final" --srcfoldername "./3seconds"
```
and
```
python preprocessdata/writellFeature_xlsx.py --srcfolderllFeature "./Final/llfeature" --groundtruth_test "./groundtruth/test_groundtruth.csv" --filenamexlsx "llFeature.xlsx"
```
# Training
Training WavemsNet
```
python trainWavemsNet.py --config_path "./config/configWavemsNet.json" --traindata "./Final/wavemsdata/train.cPickle" --validdata "./Final/wavemsdata/valid.cPickle"
```
Training llNet
```
python trainllNet.py -p "./Final/llFeature.xlsx"
```
# valdidate model
To generate the accuracy on testing dataset
```
python validateWavemsNet.py --model_path "./pretrained/model_best_46_vu.pth.tar" --testfile "./Final/wavemsdata/test.cPickle"
```