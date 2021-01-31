import json
import math, re, os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A

from functools import partial
from kaggle_datasets import KaggleDatasets

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Input, Activation, Conv2D, Add, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras_efficientnets import EfficientNetB0
from sklearn.model_selection import KFold

BASE_DIR = 'E:/KaggleChallenges/cassava-leaf-disease-classification'
TRAIN_DIR = 'E:/KaggleChallenges/cassava-leaf-disease-classification/train_images'
TEST_DIR = 'E:/KaggleChallenges/cassava-leaf-disease-classification/test_images'

sub = pd.read_csv(f'{BASE_DIR}sample_submission.csv')

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE()
GCS_PATH = KaggleDatasets().get_gcs_path('cassava-leaf-disease-classification')
GCS_PATH_AUG = KaggleDatasets().get_gcs_path('cassava-aug')
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
CLASSES = ['0', '1', '2', '3', '4']
EPOCHS = 25

TRAINING_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/train_tfrecords/ld_train*.tfrec'))
TEST_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/test_tfrecords/ld_test*.tfrec'))
AUG_FILENAME = np.array(tf.io.gfile.glob(GCS_PATH_AUG + '/cassva_aug_*.tfrec'))

def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.'.search(filename).group(1)) for filename in filenames)]
    return np.sum(n)
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)


with open(os.path.join(BASE_DIR, 'label_num_to_disease_map.json')) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
print(json.dumps(map_classes, indent = 4))

train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
train['class_name'] = train['label'].map(map_classes)

plt.figure(figsize = (8, 4))
sns.countplot(y = 'class_name', data = train);


def decode_image(image):
    image = tf.iamge.decode_jpeg(image, channels = 3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTOTUNE) # Automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(partial(read_tfrecord, labeled = labeled), num_parallel_calls = AUTOTUNE)
        return dataset
    
    
def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO) statement in the following function. This happens essentially for free on TPU.
    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.adjust_contrast(image, 1.5)
    return image, label

def get_training_dataset(file_names, augmented = False):
    dataset = load_dataset(TRAINING_FILENAMES, labeled = True)
    if augmented:
        dataset = dataset.map(data_augment, num_parallel_calls = AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    
def get_test_dataset(ordered = False):
    dataset = load_dataset(TEST_FILENAMES, labeled = False, ordered = ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

count_data_items(TRAINING_FILENAMES[0:1])


import collections

file_name = {}
for i in range(len(TRAINING_FILENAMES)):
    name = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    num = count_data_items(TRAINING_FILENAMES[i:i+1]) / BATCH_SIZE
    for _, label in get_training_dataset(TRAINING_FILENAMES[i:i+1]).take(num):
        reps = collections.Counter(label.numpy())
        for key, rep in collections.Counter(label.numpy()).most_common():
            name[str(key)] += rep
    file_name[TRAINING_FILENAMES[i]] = name
    
    
file_name

def efn():
    inputs = Inputs(shape = (*IMAGE_SIZE, 3))
    model = EfficientNetB0(include_top = False, input_tensor = inputs, weights = 'imagenet')
    model.trainable = False
    
    for layer in model.layers[-12:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
            
            
    x = GlobalAveragePooling2D(name = 'avg_pool')(model.output)
    x = BatchNormalization()(x)
    
    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name = 'top_dropout')(x)
    outputs = Dense(5, activation = 'softmax', name = 'prediction')(x)
    
    # Compile
    
    model = tf.keras.Model(inputs, outputs, name = 'EfficientNet')
    optimizer = Adam(learning_rate = 1.5e-3)
    
    model_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'spa')]
    model_loss = tf.keras.losses.SparseCategoricalCrossentropy(name = 'scc')
    model.compile(optimizer = optimizer, loss = model_loss, metrics = model_metrics)
    return model

with strategy.scope():
    model = efn()
    
model.summary()
def plot(hist):
    fig, axis = plt.subplots(len(hist), 2, figsize = (1, 0, 10))
    for ax_index, history in enumerate(hist.values()):
        history_frame = pd.DataFrame(history.history)
        history_frame.loc[:, ['loss', 'val_loss']].plot(ax=axes[ax_index][0], title = 'Losses')
        history_frame.loc[:, ['spa', 'val_spa']].plot(ax = axes[ax_index][1], title = 'Accuracies')
        

def run_kfold(model, seed = 2983, epochs=50, n_fold = 10):
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    EPOCHS = epochs
    N_FOLDS = n_fold
    kd = KFold(n_splits = N_FOLDS, random_state = seed, shuffle = True)
    hist = {}
    
    for fold_index, (train_indices, val_indices) in enumerate(kf.split(TRAINING_FILENAMES)):
        print(f'Fold {fold_index + 1} ---------------------------------------------------------------------------')
        
        train, val = get_training_dataset(TRAINING_FILENAMES[train_indices], augmented = True), get_training_dataset(TRAINING_FILENAMES[val_indices])
        STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES[train_indices]) // BATCH_SIZE
        VALID_STEPS = count_data_items(TRAINING_FILENAMES[val_indices]) // BATCH_SIZE
        hist[fold_index] = model.fit(train, 
                                     steps_per_epoch = STEPS_PER_EPOCH,
                                     epochs = EPOCHS,
                                     verbose = 1,
                                     validation_data = val,
                                     validation_steps = VALID_STEPS,
                                     callbacks = [EarlyStopping(monitor = 'val_loss',
                                                                patience = 30,
                                                                restore_best_weights = True,
                                                                min_delta = 0,
                                                                verbose = True),
                                                  ReduceLROnPlateau(monitor = 'val_spa',
                                                                    factor = 0.9,
                                                                    patience = 10,
                                                                    verbose = 0,
                                                                    mode = 'max',
                                                                    min_delta = 0.001,
                                                                    cooldown = 0,
                                                                    min_lr = 0)])
        return hist
    
    
hist = run_kfold(model, epochs = 150, n_fold = 6)

plot(hist)

imgs = np.empty((5, 600, 800, 3))

c0_name = train[train['label'] == 0].iloc[1000]['image_id']
imgs[0] = plt.imread(f'{TRAIN_DIR}{c0_name}') / 255

c1_name = train[train['label'] == 1].iloc[100]['image_id']
imgs[1] = plt.imread(f'{TRAIN_DIR}{c1_name}') / 255

c2_name = train[train['label'] == 2].iloc[230]['image_id']
imgs[2] = plt.imread(f'{TRAIN_DIR}{c2_name}') / 255

c3_name = train[train['label'] == 3].iloc[230]['image_id']
imgs[3] = plt.imread(f'{TRAIN_DIR}{c3_name}') / 255

c4_name = train[train['label'] == 4].iloc[230]['image_id']
imgs[4] = plt.imread(f'{TRAIN_DIR}{c4_name}') / 255



plt.imshow(imgs[3])



model.predict(imgs)


model.save(f'E:/KaggleChallenges/cassava-leaf-disease-classification', save_format = 'h5')

testing_dataset = get_test_dataset()


def to_float32(image, label):
    return tf.cast(image, tf.float32), label

test_ds = get_test_dataset(ordered = True)
test_ds = test_ds.map(to_float32)

print('Computing predictions...')
test_images_ds = testing_dataset
test_iamges_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axes = -1)
print(predictions)




NUM_TEST_IMAGES = len(TEST_FILENAMES)
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt = ['%s', '%d'], delimiter = ',', header = 'id, label', comments = '')
#!head submission.csv