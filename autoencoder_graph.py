from create_smiles_graph_npy import SparseMolecularDataSet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 128, 128, 1))
    return array

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape([128, 128,1]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape([128, 128,1]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('out.png')

# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
# (train_data_tgt, ), (test_data_tgt, ) = mnist.load_data()

train_data_tgt = []
train_data_src = []
train_count = 0
# for filename in glob.glob('USPTO-50K-IMAGES-TGT-TRAIN/*'):
#     train_data_tgt.append(np.reshape(np.load(filename), [128, 128, 1]))

for filename in glob.glob('USPTO-50K-IMAGES-SRC-TRAIN/*'):
    train_count += 1
    train_data_src.append(np.reshape(np.load(filename), [128, 128, 1]))

data_train = SparseMolecularDataSet()
data_train.load("./tgt_train.sparsedataset")
# data_train.log(data_train[0])
all_idx = np.random.permutation(train_count)
train_idx = all_idx[0:train_count]
train_data_tgt = data_train._next_batch(0,train_count,train_idx,train_count)
# for idx in range(train_count):
#     train_data_tgt.append(data_train._next_batch(0,train_count,idx,train_count))

test_data_tgt = []
test_data_src = []
test_count = 0
# for filename in glob.glob('USPTO-50K-IMAGES-TGT-TEST/*'):
#     test_data_tgt.append(np.reshape(np.load(filename), [128, 128, 1]))

for filename in glob.glob('USPTO-50K-IMAGES-SRC-TEST/*'):
    test_data_src.append(np.reshape(np.load(filename), [128, 128, 1]))

data_test = SparseMolecularDataSet()
data_test.load("./tgt_test.sparsedataset")
all_idx = np.random.permutation(test_count)
test_idx = all_idx[0:test_count]
test_data_tgt = data_train._next_batch(0,test_count,test_idx,test_count)
# for idx in range(test_count):
#     test_data_tgt.append(data_train._next_batch(0,test_count,idx,1))
    
# Normalize and reshape the data
train_data_tgt = np.array(train_data_tgt)
test_data_tgt = np.array(test_data_tgt)
train_data_src = np.array(train_data_src)
test_data_src = np.array(test_data_src)

# train_data_tgt = preprocess(train_data_tgt)
# test_data_tgt = preprocess(test_data_tgt)
train_data_src = preprocess(train_data_src)
test_data_src = preprocess(test_data_src)
# Create a copy of the data with added noise
# train_data_src = noise(train_data_tgt)
# test_data_src = noise(test_data_tgt)

# Display the train data and a version of it with added noise
# display(train_data_tgt, train_data_src)

input = layers.Input(shape=(128, 128, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# ADD 2 DENSE LAYERS

# ADD DENSE LAYER

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
autoencoder.summary()

# PLAY W/ LEARNING RATE & BATCH SIZE
# LR .1 .01 .001
# BATCH SIZE 20, 100, 500
autoencoder.fit(
    x=train_data_src,
    y=train_data_tgt,
    epochs=60,
    batch_size=50,
    shuffle=True,
    validation_data=(test_data_src, test_data_tgt),
)

predictions = autoencoder.predict(test_data_src)
# display(test_data_tgt, predictions)
