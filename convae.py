import numpy as np # this module is useful to work with numerical arrays
from tqdm import tqdm # this module is useful to plot progress bars
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import random_split
from torch import nn
import torch.optim as optim
from mydataset import MyDataset
from torch.utils.data import Subset

# create dataset object
train_dataset = MyDataset(idx_to_file = 'idx_to_file_train.json', data_folder = 'dataset/train')
test_dataset = MyDataset(idx_to_file = 'idx_to_file_test.json', data_folder = 'dataset/test')


# transform the input data type to torch tensor
train_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform


# split the train data to train and val
m = len(train_dataset)
train_frac = 0.5
train_data, val_data = random_split(train_dataset, [int(m*train_frac), m - int(m*train_frac)])

# training batch size
batch_size = 2

# the dataloaders handle shuffling, batching, etc...
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim),
            nn.LayerNorm(encoded_space_dim, elementwise_affine=False)
        )

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x


# set the random seed for reproducible results
torch.manual_seed(0)

# latent space dimension
d = 10

# instantiate model 
encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)

# change model parameter data type to double
encoder = encoder.double()
decoder = decoder.double()

# define the loss function
loss_fn = torch.nn.MSELoss()

# learning rate
lr = 0.001

# model parameters
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]


# oiptimizer
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# device: cpu/gpu
device = torch.device("cpu")

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    # with "_" we just ignore the labels (the second element of the dataloader tuple)
    for image_batch in dataloader:
        # Move tensor to the proper device
        image_batch = torch.unsqueeze(image_batch, 1)
        image_batch = image_batch.to(device)
        # Encode data

        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


# Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = torch.unsqueeze(image_batch, 1)
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


num_epochs = 20
history = {'train_loss': [], 'val_loss': []}
for epoch in range(num_epochs):

    train_loss = train_epoch(encoder, decoder, device,
                             train_loader, loss_fn, optim)
    val_loss = test_epoch(encoder, decoder, device, valid_loader, loss_fn)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch +
          1, num_epochs, train_loss, val_loss))
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
