


LABELS = [g + '_' + a for g in ['female', 'male']  for a in ['north', 'central', 'south']]

LABELS_MAPPING_ID = {
    "female_north": 0,
    "female_central": 1,
    "female_south": 2,
    "male_north": 3,
    "male_central": 4,
    "male_south": 5
}


def assignlabel_id(row):
    # print(row[0])
    if row.gender == 0:
        if row.accent == 0: return 0
        elif row.accent == 1: return 1
        else: return 2
    else:
        if row.accent == 0: return 3
        elif row.accent == 1: return 4
        else: return 5

def convertwav(row):
    name = row.id.split(".")[0]
    return name+".wav"