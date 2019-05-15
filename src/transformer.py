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
    pitch = np.linspace(note_to_freq('C4'), note_to_freq('C5'), 80) / note_to_freq('C8')
    volume = np.random.uniform(0.001, 0.5)
    singer = np.random.randint(20)
    vowel = np.random.randint(5)
    x = np.zeros((27, 80), dtype=np.float32)
    x[0] = pitch
    x[1] = volume
    x[2 + singer] = 1
    x[22 + vowel] = 1
    x = torch.unsqueeze(torch.tensor(x), 0)

    return x

def train():
    #simple training process
    model_directory = os.path.join('.', 'models', 'vocal_transformer')

    model = reformer() #transformer(batch_norm=True)
    loader = VocalSetLoader(duration=1)
    train_loader = DataLoader(loader, num_workers=1, shuffle=True, batch_size=200)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    i = 0 #current iteration
    for epoch in range(1000):

        print('Epoch %d -------------------------' % epoch)
        if epoch % 10 == 0:
            #save model/optimizer
            print('Saving model at epoch %d' % epoch)
            torch.save(model.state_dict(), os.path.join(model_directory, ('model_%d' % epoch)))
            torch.save(optimizer.state_dict(), os.path.join(model_directory, ('optim_%d' % epoch)))

        for batch in train_loader:
            model.zero_grad()
            x, y = batch
            y_hat = model(x)

            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            print('%d:  %f' % (i, loss.data.item()))
            with open(os.path.join(model_directory, 'loss_history.txt'), 'a') as f:
                f.write('%s\n' % str(loss.data.item()))

            i += 1 #increment iteration number


def infer():
    #simple inference to check results
    model_directory = os.path.join('.', 'models', 'vocal_transformer')


    model = transformer(batch_norm=True)
    model.load_state_dict(torch.load(os.path.join(model_directory, 'model_430')))
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


def synthesize():
    #create spectrograms from scratch
    model_directory = os.path.join('.', 'models', 'vocal_transformer', '6_layer_with_batch_norm')
    model = transformer(batch_norm=True)
    model.load_state_dict(torch.load(os.path.join(model_directory, 'model_430')))
    model.eval()

    for i in range(10):
        x = generate_features()
        y_hat = model(x).detach()

        torch.save(y_hat[0], os.path.join(model_directory, 'spec_%d.pt' % i))





if __name__ == '__main__':
    train()
    #infer()
    #synthesize()