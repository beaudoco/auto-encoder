import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
from create_smiles_test_graph_npy import SparseMolecularTestDataSet
from create_smiles_train_graph_npy import SparseMolecularTrainDataSet

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 128, 128, 1))
    return array

n_epochs = 30   # Number of optimization epochs
n_layers = 1    # Number of random layers
n_train = 50    # Size of the train dataset
n_test = 30     # Size of the test dataset

SAVE_PATH = "graph/" # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_random_seed(0)       # Seed for TensorFlow random number generator

train_data_tgt = []
train_data_src = []


for filename in glob.glob('USPTO-50K-IMAGES-SRC-TRAIN/*'):
    train_data_src.append(np.reshape(np.load(filename), [128, 128, 1]))

train_count = len(train_data_src)
data_train = SparseMolecularTrainDataSet()
data_train.load("./tgt_train.sparsedataset")
all_idx = np.random.permutation(train_count)
train_idx = all_idx[0:train_count]
train_data_tgt = data_train._next_batch(0,train_count,train_idx, None)

test_data_tgt = []
test_data_src = []

for filename in glob.glob('USPTO-50K-IMAGES-SRC-TEST/*'):
    test_data_src.append(np.reshape(np.load(filename), [128, 128, 1]))

test_count = len(test_data_src)
data_test = SparseMolecularTestDataSet()
data_test.load("./tgt_test.sparsedataset")
all_idx = np.random.permutation(test_count)
test_idx = all_idx[0:test_count]
test_data_tgt = data_test._next_batch(0,test_count,test_idx, None)

# print(len(test_data_tgt))

train_images = np.array(train_data_src)
train_labels = train_data_tgt
test_images = np.array(test_data_src)
test_labels = test_data_tgt

# mnist_dataset = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# print(train_images[0].shape)

# Normalize pixel values within 0 and 1
# train_images = train_images / 255
# test_images = test_images / 255
train_images = preprocess(train_images)
test_images = preprocess(test_images)

# Add extra dimension for convolution channels
train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

dev = qml.device("default.qubit", wires=4)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((64, 64, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 128, 2):
        for k in range(0, 128, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k, 0, 0],
                    image[j, k + 1, 0, 0],
                    image[j + 1, k, 0, 0],
                    image[j + 1, k + 1, 0, 0]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, ..., c] = q_results[c]
    return out

if PREPROCESS == True:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        # print(img.shape)
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)
    
    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


# Load pre-processed images
q_train_images = np.array(np.load(SAVE_PATH + "q_train_images.npy"))
q_test_images = np.array(np.load(SAVE_PATH + "q_test_images.npy"))

print(q_train_images.shape)
for x in q_train_images:
    print(x)

# print(q_train_images.shape)
# quit()

n_samples = 4
n_channels = 4
fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
    axes[0, 0].set_ylabel("Input")
    if k != 0:
        axes[0, k].yaxis.set_visible(False)
    axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

    # Plot all output channels
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
        if k != 0:
            axes[c, k].yaxis.set_visible(False)
        print(q_train_images.shape)
        axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

plt.tight_layout()
plt.show()

def MyModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

q_model = MyModel()

q_history = q_model.fit(
    q_train_images,
    train_labels,
    validation_data=(q_test_images, test_labels),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

c_model = MyModel()

c_history = c_model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(q_history.history["val_acc"], "-ob", label="With quantum layer")
ax1.plot(c_history.history["val_acc"], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], "-ob", label="With quantum layer")
ax2.plot(c_history.history["val_loss"], "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()

