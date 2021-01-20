import argparse
import os
from imutils import paths

import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile

from config import config
from utils import chunk

from multiprocessing import cpu_count, Pool
from itertools import product


LABELS = config.LABELS

def extract_low_feature(segment=None, sampling_rate=16000):

    """Extract global feature of N-duration audio
    Args:
        segment (array): N-duration audio
        sampling_rate (int): `segment` sampling rate
        dst_filename (str): path to store the resulted image
    """
    # Zero - crossing rate:
    zrc_mean = np.mean(librosa.feature.zero_crossing_rate(segment).T, axis=0)

    # root mean square energy
    rmse_mean = np.mean(librosa.feature.rms(segment).T, axis=0)

    # Compute Short - Time Fourier Transform
    stft = np.abs(librosa.stft(segment))

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = librosa.feature.mfcc(y=segment, sr=sampling_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sampling_rate).T,axis=0)

    # Computes spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sampling_rate).T, axis=0)

    # Computes spectral rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(segment, sr=sampling_rate).T, axis=0)

    # Computes spectral bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(segment, sr=sampling_rate).T, axis=0)

    # computes spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(segment).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T,axis=0)

    # Computes spectral flux
    flux = librosa.onset.onset_strength(segment, sr=sampling_rate)
    flux_mean = np.array([flux.mean()])

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(segment), \
        sr=sampling_rate).T,axis=0)
    
    return [zrc_mean, rmse_mean, mfccs_mean, chroma, mel, centroid, rolloff, bandwidth, flatness, contrast, flux_mean, tonnetz]

def build_extract_feature(datainfo):
    # samplingrate = int(samplingrate)
    # print(samplingrate)
    dstfoldername = "./Final"
    dstfolder = os.path.join(dstfoldername, datainfo['feature_name'], datainfo['label'])

    for filename in datainfo['path']:
        np_path = os.path.join(dstfolder, filename[filename.rfind(os.path.sep) + 1:1 + filename.rfind('.')] + 'npz')

        if os.path.isfile(np_path): continue
        
        # extract feature
        audio, _ = librosa.load(filename, sr=16000)
        llfeature = np.concatenate(extract_low_feature(audio, sampling_rate=16000))
        np.save(np_path, llfeature)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--feature_name', required=True, type=str, help='Name of feature to be extracted')
    ap.add_argument('-p', '--procs', default=-1, type=int, help='Number of processes to launch')
    # ap.add_argument('-sr', '--sampling_rate', type=str, help='Sampling rate of signal')
    ap.add_argument('--dstfoldername', required=True, type=str, help="Folder name of extrated feature")
    ap.add_argument('--srcfoldername', required=True, type=str, help="Folder name of audio file")

    args = vars(ap.parse_args())

    args['feature_name'] = args['feature_name'].lower()
    
    procs = args['procs'] if args['procs'] > 0 else cpu_count()

    # convert training set and public test set
    for label in LABELS + ['public_test']:
        print(f'Converting to npy for {label}....')
        src_folder = os.path.join(args['srcfoldername'], label)
        dst_folder = os.path.join(args['dstfoldername'], args['feature_name'], label)

        if not os.path.isdir(dst_folder): os.makedirs(dst_folder)

        file_paths = list(paths.list_files(src_folder))
        num_files_per_proc = len(file_paths) // procs
        chunked_paths = list(chunk(file_paths, num_files_per_proc))

        datainfo = []
        for label, chunkedpath in zip([label] * num_files_per_proc, chunked_paths):
            chunkedinfo = {
                'feature_name': args['feature_name'],
                'label': label,
                'path': chunkedpath
            }
            datainfo.append(chunkedinfo)

        pool = Pool(processes=procs)
        pool.map(build_extract_feature, datainfo)
