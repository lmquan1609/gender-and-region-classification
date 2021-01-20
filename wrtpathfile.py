from config import config
import os
import argparse
import random
import pandas as pd

LABELS_MAPPING_ID = config.LABELS_MAPPING_ID
LABELS = config.LABELS

def main(audiopath, pathofaudiofile_folder, metaFile, groundtruthtest):
    
    print("Process create file  contains path train audio")
    if not os.path.exists(pathofaudiofile_folder):
        os.makedirs(pathofaudiofile_folder)

    f = open(metaFile, "w")

    for label in LABELS:
        absDirection = os.path.join(audiopath, label)
        classID = LABELS_MAPPING_ID[label]

        for audiofile in os.listdir(absDirection):
            f.writelines(label + '/' + audiofile + '\t' + str(classID) + '\n')
        
    f.close()

    print("Process create file txt contains path audio")
    dataframeOftest = pd.read_csv(groundtruthtest)
    # label = dataframeOftest.apply(config.assignlabel, axis=1)
    labelid = dataframeOftest.apply(config.assignlabel_id, axis=1)
    filename_wav = dataframeOftest.apply(config.convertwav, axis=1)

    # dataframeOftest['label'] = pd.DataFrame(label)
    dataframeOftest['labelid'] = pd.DataFrame(labelid)
    dataframeOftest['filename_wav'] = pd.DataFrame(filename_wav)
    data = dataframeOftest[['filename_wav', 'labelid']]

    with open(os.path.join(pathofaudiofile_folder, 'test.txt'), 'w') as f:
        for i in range(len(data)):
            f.write("public_test" + '/' + data.iloc[i]['filename_wav'] + '\t' + str(data.iloc[i]['labelid']) + '\n')

    print("-----Split train and valid -----")

    f = open(metaFile)
    lines = f.readlines()

    validlist = random.sample(lines, int(0.2*len(lines)))
    trainlist = list(set(lines)^set(validlist))

    for namelist, filename in zip([trainlist, validlist],['train.txt', 'valid.txt']):
        with open(os.path.join(pathofaudiofile_folder, filename), 'w') as f:
            for item in namelist:
                f.write(item)
    print("DONE")

    

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--audiopath', required=True, type=str, help='Audio path')
    ap.add_argument('--foldername', required=True, type=str, help='Create Folder contain path of audio file')
    ap.add_argument('--metafile', required=True, type=str, help='File write path audio')
    ap.add_argument('--labeltest', required=True, type=str, help='File ground truth of test data')
    args = vars(ap.parse_args())

    main(args['audiopath'], args['foldername'], args['metafile'], args['labeltest'])

