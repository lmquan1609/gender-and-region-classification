LABELS = [g + '_' + a for g in ['female', 'male']  for a in ['north', 'central', 'south']]

MAPPINGS = {
    'female_north': 0,
    'female_central': 1, 
    'female_south': 2,
    'male_north': 3,
    'male_central': 4,
    'male_south': 5
}

INVERSE_MAPPINGS = {
    0: 'female_north',
    1: 'female_central', 
    2: 'female_south',
    3: 'male_north',
    4: 'male_central',
    5: 'male_south'
}

GENDER_MAPPINGS = {
    0: 'female',
    1: 'male'
}

ACCENT_MAPPINGS = {
    0: 'north',
    1: 'central',
    2: 'south'
}

TEST_GT = './groundtruth/test_groundtruth.csv'
CORRECT_CLASSIFICATION_PATH = 'correct_classification.csv'
INCORRECT_CLASSIFICATION_PATH = 'incorrect_classification.csv'

INPUT_DIMS = [3, 128, 250]

NUM_CLASSES = 6