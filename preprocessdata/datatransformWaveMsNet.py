import librosa
import random
from config import config
import os
import argparse


def main(audiofolder, samplingrate, traintxt, validtxt, testtxt, folderpickle):
    """
    :return  [{'key', 'data', 'label'}, ... ]
    """
    # wav_len = fs * 3

    if not os.path.exists(folderpickle):
        os.makedirs(folderpickle)

    trainWaveList = config.get_wavelist(traintxt)
    validWaveList = config.get_wavelist(validtxt)
    testWaveList = config.get_wavelist(testtxt)


    waveLists = [trainWaveList, validWaveList, testWaveList]

    data = []
    item = {}

    for idx, wavelist in enumerate(waveLists):
        for f in wavelist:
            cls_id = f.split('\t')[1]
            audio, _ = librosa.load(os.path.join(audiofolder, f.split('\t')[0]), samplingrate)
    
            item['label'] = int(cls_id)
            item['key'] = f.split('/')[1].split('.')[0]
            item['data'] = audio

            data.append(item)
            item = {}

        if idx == 0:
            print("Create pickle file for training set")
            random.shuffle(data)
            config.save_data(folderpickle + "/train.cPickle", data)
        
        elif idx == 1:
            print("Create pickle file for validation set")
            random.shuffle(data)
            config.save_data(folderpickle + "/valid.cPickle", data)

        else: 
            print("Create pickle file for test set")
            random.shuffle(data)
            config.save_data(folderpickle + "/test.cPickle", data)
        
        data = []

    
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--audiofolder', required=True, type=str, help="Audio folder data")
    ap.add_argument('--samplingrate', required=True, type=int, help="Sampling rate of audio signal")
    ap.add_argument('--traintxt', required=True, type=str, help="File train txt")
    ap.add_argument('--validtxt', required=True, type=str, help="File valid txt")
    ap.add_argument('--testtxt', required=True, type=str, help="File test txt")
    ap.add_argument('--folderpickle', required=True, type=str, help="Folder of pickle fie")

    args = vars(ap.parse_args())

    main(args['audiofolder'], args['samplingrate'], args['traintxt'], args['validtxt'], args['testtxt'], args['folderpickle'])