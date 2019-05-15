#simple network to convert a series of pitch/volume points into a mel-spectrogram for the wavenet to perform inference on
#loosely modeled after the U-net from assignment 6
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

from math import ceil, floor
from vocalset_loader import VocalSetLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pure_waves import pitch as note_to_freq
import os
import re

import pdb



def add_conv(in_channels, out_channels, kernel_size=3, stride=1, batch_norm=False):
    padding = floor(kernel_size/2) #pad so that same size input
    
    if batch_norm:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

class transformer(nn.Module):
    def __init__(self, in_channels=27, out_channels=80, batch_norm=False):
        super(transformer, self).__init__()

        #create conv_stages
        self.conv1 = add_conv(in_channels, 30, batch_norm=batch_norm)
        self.conv2 = add_conv(30, 45, batch_norm=batch_norm)
        self.conv3 = add_conv(45, 60, batch_norm=batch_norm)
        self.conv4 = add_conv(60, out_channels, batch_norm=batch_norm)
        self.conv5 = add_conv(out_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x *= -1 #nv-wavenet spectograms have only negative values, so invert network output to get the negative-only
        return x

def add_conv_stage(in_channels, out_channels, kernel_size=3, stride=1):
    padding = floor(kernel_size/2) #pad so that same size input
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(), #suspect LeakyReLU isn't needed because output is only positive
        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class reformer(nn.Module):
    """simple residual transformer"""
    def __init__(self, in_channels=27, out_channels=80):
        super(reformer, self).__init__()

        self.conv1 = add_conv_stage(in_channels, out_channels)
        self.conv2 = add_conv_stage(out_channels + in_channels, out_channels)
        self.conv3 = add_conv_stage(out_channels + in_channels, out_channels)
        self.conv4 = add_conv_stage(out_channels + in_channels, out_channels)
        self.conv5 = add_conv_stage(out_channels + in_channels, out_channels)
        self.conv6 = add_conv_stage(out_channels + in_channels, out_channels)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        a = x

        a = self.conv1(a)
        a = torch.cat((x, a), 1)
        a = self.dropout(a)
        
        a = self.conv2(a)
        a = torch.cat((x, a), 1)
        a = self.dropout(a)

        a = self.conv3(a)
        a = torch.cat((x, a), 1)
        a = self.dropout(a)

        a = self.conv4(a)
        a = torch.cat((x, a), 1)
        a = self.dropout(a)

        a = self.conv5(a)
        a = torch.cat((x, a), 1)
        a = self.dropout(a)

        a = -1 * self.conv6(a) #final layer, use output directly (negative because spectrogram is negative only)

        return a




def generate_features():
    pitch = np.concatenate([
        np.ones(20) * note_to_freq('C4') / note_to_freq('C8'),
        np.linspace(note_to_freq('C4'), note_to_freq('C5'), 80) / note_to_freq('C8'),
        np.ones(80) * note_to_freq('C5') / note_to_freq('C8'),
        np.linspace(note_to_freq('C5'), note_to_freq('C4'), 80) / note_to_freq('C8'),
        np.ones(20) * note_to_freq('C4') / note_to_freq('C8')
    ])
    volume = np.random.uniform(0.001, 0.5)
    singer = np.random.randint(20)
    vowel = np.random.randint(5)
    x = np.zeros((27, pitch.shape[0]), dtype=np.float32)
    x[0] = pitch
    x[1] = volume
    x[2 + singer] = 1
    x[22 + vowel] = 1
    x = torch.unsqueeze(torch.tensor(x), 0)

    return x


def get_newest_idx(model_directory):
    epochs = [int(filename[len('model_'):]) for filename in os.listdir(model_directory) if re.match(r'model_[0-9]+', filename) is not None]
    return max(epochs)

def load_model(model_directory, load_idx, model, optimizer=None):
    model.load_state_dict(torch.load(os.path.join(model_directory, 'model_%d' % load_idx)))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(os.path.join(model_directory, 'optim_%d' % load_idx)))
    
    with open(os.path.join(model_directory, 'checkpoint.txt'), 'r') as f:
        nums = [int(line) for line in f]
        start_epoch = nums[0]
        i = nums[1]
    
    print('Loaded model at epoch %d and iteration %d' % (start_epoch, i))
    return model, optimizer, start_epoch, i

def save_model(model_directory, epoch, i, model, optimizer, loss_history):
    #save model/optimizer
    print('Saving model at epoch %d and iteration %d' % (epoch, i))
    
    torch.save(model.state_dict(), os.path.join(model_directory, ('model_%d' % epoch)))
    torch.save(optimizer.state_dict(), os.path.join(model_directory, ('optim_%d' % epoch)))
    
    with open(os.path.join(model_directory, 'checkpoint.txt'), 'w') as f:
        f.write('%s\n' % str(epoch))
        f.write(str(i))
    
    with open(os.path.join(model_directory, 'loss_history.txt'), 'a') as f:
        for loss in loss_history:
            f.write('%s\n' % str(loss))


def train(model_directory, load_idx=None):
    #simple training process

    model = reformer() #transformer(batch_norm=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    i = 0 #current iteration

    if load_idx is not None:
        model, optimizer, start_epoch, i = load_model(model_directory, load_idx, model, optimizer)

    model.train()
    
    loader = VocalSetLoader(duration=1)
    train_loader = DataLoader(loader, num_workers=1, shuffle=True, batch_size=2000)
    loss_history = []

    for epoch in range(start_epoch, 10000000):

        print('Epoch %d -------------------------' % epoch)
        if epoch != start_epoch and epoch % 10 == 0:
            save_model(model_directory, epoch, i, model, optimizer, loss_history)
            loss_history.clear()

        for batch in train_loader:
            model.zero_grad()
            x, y = batch
            y_hat = model(x)

            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            print('%d:  %f' % (i, loss.data.item()))
            loss_history.append(loss.data.item())

            i += 1 #increment iteration number


def infer(model_directory, load_idx):
    #simple inference to check results

    model = reformer()#transformer(batch_norm=True)
    model.load_state_dict(torch.load(os.path.join(model_directory, 'model_%d' % load_idx)))
    model.eval()

    loader = VocalSetLoader(duration=1)
    test_loader = DataLoader(loader, num_workers=1, shuffle=True, batch_size=200)

    for batch in test_loader:
        x, y = batch
        y_hat = model(x).detach()

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(y[0])
        ax2.imshow(y_hat[0])
        plt.show()


def synthesize(model_directory, load_idx):
    #create spectrograms from scratch
    model = reformer()#transformer(batch_norm=True)
    model.load_state_dict(torch.load(os.path.join(model_directory, 'model_%d' % load_idx)))
    model.eval()

    for i in range(10):
        x = generate_features()
        y_hat = model(x).detach()

        plt.imshow(y_hat[0])
        plt.show()
        torch.save(y_hat[0], os.path.join(model_directory, 'spec_%d.pt' % i))





if __name__ == '__main__':
    model_directory = os.path.join('.', 'models', 'vocal_transformer')
    # model_directory = os.path.join('.', 'models', 'vocal_transformer', '6_layer_with_batch_norm')

    load_idx = get_newest_idx(model_directory)
    train(model_directory, load_idx)
    #infer(model_directory, load_idx)
    #synthesize(model_directory, load_idx)