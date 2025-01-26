from imports import *

data_dir = './bird_dataset/'
train_path = os.path.join(data_dir,'train')
test_path = os.path.join(data_dir,'test')
val_path = os.path.join(data_dir,'val')



IMG_SIZE = (400, 400)
SEED = 50
BATCH_SIZE = 32

# Building training data with appropriate tf parameters
train_data = tf.keras.utils.image_dataset_from_directory(train_path,
                                                         image_size = IMG_SIZE,
                                                         label_mode = 'categorical',
                                                         batch_size = BATCH_SIZE,
                                                         shuffle = True,
                                                         seed = SEED)

# Building testing data with appropriate tf parameters
test_data = tf.keras.utils.image_dataset_from_directory(test_path,
                                                        image_size=IMG_SIZE,
                                                        label_mode='categorical',
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)

# Building val data with appropriate tf parameters
validation_data = tf.keras.utils.image_dataset_from_directory(val_path,
                                                              image_size=IMG_SIZE,
                                                              label_mode='categorical',
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              seed=SEED)
