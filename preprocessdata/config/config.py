import pickle

LABELS = [g + '_' + a for g in ['female', 'male']  for a in ['north', 'central', 'south']]

LABELS_MAPPING_ID = {
    "female_north": 0,
    "female_central": 1,
    "female_south": 2,
    "male_north": 3,
    "male_central": 4,
    "male_south": 5
}

def get_wavelist(fold_wavelist):
    f = open(fold_wavelist, 'r')
    waveList = []
    for line in f.readlines():
        waveList.append(line)
    return waveList

def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))