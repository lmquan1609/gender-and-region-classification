import pandas as pd
from config import config
import numpy as np
from imutils import paths
from natsort import natsorted
import argparse
import os

def assignlabel(row):
    # print(row[0])
    if row.gender == 0:
        if row.accent == 0: return "female_north"
        elif row.accent == 1: return "female_central"
        else: return "female_south"
    else:
        if row.accent == 0: return "male_north"
        elif row.accent == 1: return "male_central"
        else: return "male_south"

LABELS = config.LABELS

def main(srcfolder, groundtruth_test, filenamexlsx):

    ll_feature_train_data, ll_feature_train_label = [], []
    ll_feature_test_data = []

    for label in LABELS:
        src_folder = os.path.join(srcfolder, label)
        for path in paths.list_files(src_folder):
            ll_feature_train_data.append(np.load(path))
            ll_feature_train_label.append(label)

    ll_feature_train_data = np.vstack(ll_feature_train_data)

    df = pd.DataFrame(ll_feature_train_data)
    df['label'] = pd.DataFrame(ll_feature_train_label)

    src_folder = os.path.join(src_folder, "public_test")
    for path in natsorted(paths.list_files(src_folder)):
        ll_feature_test_data.append(np.load(path))

    ll_feature_test_data = np.vstack(ll_feature_test_data)

    ll_feature_test_label = pd.read_csv(groundtruth_test)
    label = ll_feature_test_label.apply(assignlabel, axis=1)

    df_test = pd.DataFrame(ll_feature_test_data)
    df_test['label'] = pd.DataFrame(label)

    with pd.ExcelWriter(os.path.join(srcfolder, filenamexlsx)) as writer:
        df.to_excel(writer, sheet_name="train_llFeature")
        df_test.to_excel(writer, sheet_name="test_llFeature")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcfolderllFeature', required=True, type=str, help="Source folder low level feature")
    ap.add_argument('--groundtruth_test', required=True)
    ap.add_argument('--filenamexlsx', required=True)
    args = vars(ap.parse_args())

    main(args['srcfolderllFeature'], args['groundtruth_test'], args['filenamexlsx'])