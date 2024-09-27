import os
import shutil
from glob import glob

import tensorflow as tf
import numpy as np
from .model import *

CHUNK_SIZE = 40960
INPUT_PATH='/kaggle/input'
WORKING_PATH='/kaggle/working'
SYMLINK='kaggle'

shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(WORKING_PATH, 0o777, exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f'setting SEED to {seed}')

class Config:
    IMG_W =  512
    IMG_H = 512
    RESIZE_H = 700
    RESIZE_W = 1200
    LAMBDA = 10
    BUFFER_SIZE = 100
    BATCH_SIZE = 2
    CACHE= 50
    LR = 0.00025
    SEED = 7

set_seed(Config.SEED)

day_dir = '../input/daynight-cityview/day'
night_dir = '../input/daynight-cityview/night'

def loadImage(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    return image

def crop(image):
    img = tf.image.random_crop(image, size=[Config.IMG_H, Config.IMG_W, 3])
    return img


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def deNormalize(image):
    return (image * 0.5) + 0.5

def img_aug(image):
    rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if rotate > .8:
        image = tf.image.rot90(image, k=3) 
    elif rotate > .6:
        image = tf.image.rot90(image, k=2)
    elif rotate > .4:
        image = tf.image.rot90(image, k=1) 

    flip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if flip > 0.7:
        image = tf.image.random_flip_left_right(image)
    elif flip < 0.3:
        image = tf.image.random_flip_up_down(image)

    return image

def jitter(image):
    image = tf.image.resize(image, size=(Config.RESIZE_H, Config.RESIZE_W), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = crop(image)
    return image


def prepTrain(image):
    image = loadImage(image)
    image = jitter(image)
    image= img_aug(image)
    image = normalize(image)
    return image

def prepEval(image):
    image = loadImage(image)
    image = jitter(image)
    image = normalize(image)
    return image

def imgDataset(directory,
                       prep,
                       image_extension = 'jpg',
                       repeat=True
                      ):
    images = glob(directory+f'/*{image_extension}')
    dataset = tf.data.Dataset.list_files(images)

    dataset = dataset.map(prep,
                          num_parallel_calls=tf.data.AUTOTUNE)

    if repeat :
        dataset = dataset.repeat()

    dataset = dataset.shuffle(Config.BUFFER_SIZE)
    dataset = dataset.batch(Config.BATCH_SIZE)
    return dataset


dataset_day = imgDataset(directory = day_dir,prep = prepTrain)
eval_day = imgDataset(directory = day_dir, prep = prepEval)

dataset_night = imgDataset(directory = night_dir,prep = prepTrain)
eval_night = imgDataset(directory = night_dir, prep = prepEval)

train_dataset = tf.data.Dataset.zip((dataset_day,dataset_night))

gan = get_gan(dataset_day)

gan.compile(gen_loss_function=gen_loss,
            disc_loss_function=disc_loss,
            cycle_loss_function=cycLoss,
            loss_function=identity_loss)

def scheduler(epoch,
              lr,
              decay_rate = 0.05,
              warm_up_period = 10):

    if epoch < warm_up_period:
        return lr
    elif (epoch > warm_up_period and epoch<40):
        return lr * tf.math.exp(decay_rate)
    else:
        return lr * tf.math.exp(decay_rate*2)

LR_sch = tf.keras.callbacks.LearningRateScheduler(scheduler,
                                                        verbose = 0)

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.lossN2D = np.Inf
        self.lossD2N = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        lossN2D=logs.get('gen_N2D_loss')
        lossD2N=logs.get('gen_D2N_loss')
        if (np.less(lossN2D, self.lossN2D) and np.less(lossD2N, self.lossD2N)):
            self.lossD2N = lossD2N
            self.lossN2D = lossN2D
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self,
                 num_img=1,
                 day_paths='generated_day',
                 night_paths='generated_night'):
        self.num_img = num_img
        self.day_paths = day_paths
        self.night_paths = night_paths

        if not os.path.exists(self.day_paths):
            os.makedirs(self.day_paths)

        if not os.path.exists(self.night_paths):
            os.makedirs(self.night_paths)

#     def on_epoch_end(self, epoch, logs=None):
#         for i, img in enumerate(eval_day.take(self.num_img)):


#             prediction = day2night_gen(img, training=False)[0].numpy()
#             prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
#             prediction = PIL.Image.fromarray(prediction)
#             prediction.save(f'{self.night_paths}/generated_{i}_{epoch+1}.png')

#         for i, img in enumerate(eval_night.take(self.num_img)):
#             prediction = night2day_gen(img, training=False)[0].numpy()
#             prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
#             prediction = PIL.Image.fromarray(prediction)
#             prediction.save(f'{self.day_paths}/generated_{i}_{epoch+1}.png')

EPOCHS = 75
callbacks = [LR_sch,GANMonitor(), CustomEarlyStopping(patience = 10)]
steps_per_epoch = 200

history = gan.fit(train_dataset,
                epochs = EPOCHS,
                steps_per_epoch=steps_per_epoch,
                callbacks = callbacks)
