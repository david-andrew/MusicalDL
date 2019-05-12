"""
    MusicalDL Final Project
    File: train.py
    Purpose: run the training process for the synthesizer

    @author David Samson (copied from nv-wavenet/pytorch/train.py)
    @version 1.0
    @date 2019-05-11
"""

import pdb

import __init__
import json
import os
import sys
# import time
import torch
import re

from torch.utils.data import DataLoader
from wavenet import WaveNet
from loader import SimpleWaveLoader
from utils import to_gpu

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = wavenet_config["n_out_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return torch.nn.CrossEntropyLoss()(inputs, targets)

def find_checkpoint(model_directory):
    """return the path to the newest checkpoint, or None if no checkpoints found"""
    checkpoints = [int(file[len('checkpoint_'):]) for file in os.listdir(model_directory) if re.match(r'checkpoint_[0-9]+', file) is not None]
    if len(checkpoints) > 0:
        return os.path.join(model_directory, 'checkpoint_' + str(max(checkpoints)))
    else:
        return None

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    # model_for_loading = checkpoint_dict['model']
    # model.load_state_dict(model_for_loading.state_dict())
    model = torch.load(checkpoint_path)['model'].cuda()

    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveNet(**wavenet_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train(model_directory, epochs, learning_rate,
          epochs_per_checkpoint, batch_size, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
   
    criterion = CrossEntropyLoss()
    model = WaveNet(**wavenet_config).cuda()
    # model.upsample = torch.nn.Sequential() #replace the upsample step with no operation as we manually control samples
    # model.upsample.weight = None
    # model.upsample.bias = None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration = 0
    checkpoint_path = find_checkpoint(model_directory)
    if checkpoint_path is not None:
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = SimpleWaveLoader()
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=False, drop_last=True)
    
    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):

            model.zero_grad()
            
            x, y = batch
            x = to_gpu(x).float()
            y = to_gpu(y)
            x = (x, y)  # auto-regressive takes outputs as inputs

            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_loss = loss.data.item()
            loss.backward()
            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))

            iteration += 1
            torch.cuda.empty_cache()

        if (epoch != 0 and epoch % epochs_per_checkpoint == 0):
            checkpoint_path = os.path.join(model_directory, 'checkpoint_%d' % iteration)
            save_checkpoint(model, optimizer, learning_rate, iteration,
                            checkpoint_path)
                 
            

if __name__ == "__main__":
    model_directory = sys.argv[1]
    config_path = os.path.join(model_directory, 'config.json')

    # Parse configs.  Globals nicer in this case
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    
    train_config = config["train_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config 
    wavenet_config = config["wavenet_config"]

    #these need to be left alone, even though they're superceded and basically ignored. TODO->remove this dependancy
    global dta_config                   
    data_config = config["data_config"]
   
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(model_directory, **train_config)
