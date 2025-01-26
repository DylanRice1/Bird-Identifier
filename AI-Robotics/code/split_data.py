from imports import *

bird_images = './withBackground'

SEED = 50

splitfolders.ratio(bird_images, output = 'bird_dataset', seed = SEED, ratio = (.7, .15, .15), group_prefix = None)
# splitfolders.ratio(data, output = 'UsableData', seed = SEED, ratio = (.8, .1, .1), group_prefix = None)