#debug
import pdb

#stuff
import __init__
import os
import json
import numpy as np
import torch
from math import floor, ceil
from pure_waves import pitch

#ML models
import transformer
import nv_wavenet
import utils

#simple file loader
import tkinter
from tkinter import filedialog

#plotting
from matplotlib import pyplot as plt



class demo_manager():
    def __init__(self, section_size=1):
        self.FS = 16000
        self.SS = 80
        self.section_size = section_size #number of singers per each section size

        #list which singer indices map to which voice types
        self.singers =  {
            'Soprano':  [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'Alto':     [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'Tenor':    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Baritone': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Bass':     [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        }

        self.vowels = {
            'a': 0,
            'e': 1,
            'i': 2,
            'o': 3,
            'u': 4
        }

        # self.reformer_directory = os.path.join('.', 'models', 'vocal_transformer')
        # self.reformer_checkpoint = 'model_7400'
        # self.wavenet_directory = os.path.join('.', 'models', 'WaveNet_vocoder')
        # self.wavenet_checkpoint = 'wavenet_72700'
        self.reformer_path = os.path.join('.', 'models', 'vocal_transformer', 'model_7400')
        self.wavenet_path = os.path.join('.', 'models', 'WaveNet_vocoder', 'wavenet_72700')

        self.select_recipe()
        self.compute_duration()
        self.load_reformer()
        self.load_wavenet()
        self.perform() #create the entire performance


        # self.choir_track = np.zeros(self.num_samples, dtype=np.float32) #array to store the entire ensemble performance



    def select_recipe(self):
        #get the path to the song
        tkinter.Tk().withdraw()
        self.file_path = filedialog.askopenfilename(
            initialdir='./Ensemble/recipes/', 
            title='Open Song Recipe', 
            filetypes = (("JSON Recipe","*.json"),)
        )
        if self.file_path is ():
            raise Exception("You must specify a file to continue")

        self.song_name = os.path.split(self.file_path)[1].split('.')[0]

        #dump the recipe into a dictionary
        with open(self.file_path, 'r') as f:
            self.song = json.load(f)


    def get_voice_type(self, part_name):
        """basically convert something like "Soprano II" to just "soprano" """
        types = ['Soprano', 'Alto', 'Tenor', 'Baritone', 'Bass']
        for t in types:
            if t.lower() in part_name.lower():
                return t
        raise Exception('Unrecognized voice part "' + part_name + '"')


    def compute_duration(self):
        #count the total length of the song
        self.duration = 0
        for singer, part in self.song.items():
            for note in part:
                self.duration += note['duration']
            break #stop after the first singer

        self.num_samples = ceil(self.SS * self.duration) #total number of samples


    def load_reformer(self):
        #model to create spectrograms from scratch (desired pitch/volume)
        self.reformer = transformer.reformer()
        self.reformer.load_state_dict(torch.load(self.reformer_path))
        self.reformer.eval()

        # x = generate_features()
        # y_hat = model(x).detach()

    def load_wavenet(self):
        self.wavenet = torch.load(self.wavenet_path)['model']
        self.nv_wavenet = nv_wavenet.NVWaveNet(**(self.wavenet.export_weights()))

    def generate_performance_features(self):
        features = []
        for singer, part in self.song.items():
            for i in range(self.section_size):
                features.append(self.generate_part_features(singer, part))
        
        return features

    def generate_part_features(self, singer, part):
        singer = self.get_voice_type(singer)
        singer_idx = np.random.choice(self.singers[singer])

        part_features = np.zeros((27, self.num_samples), dtype=np.float32) #27 channels for the feature data
        part_features[2+singer_idx] = 1 #set selcted singer for feature vector

        start = 0
        for note in part:
            stop = start + ceil(note['duration'] * self.SS)        
            if note['volume'] != 0:
                part_features[0, start:stop] = note['pitch'] / pitch('C8')
                part_features[1, start:stop] = note['volume'] / 3
                part_features[22] = 1 #set the vowel spoken to Ah (probably change later)
            #else leave as all zeros
            start = stop #update current index in part

        return part_features

    def generate_performance_spectrograms(self, features):
        features = np.stack(features, 0)
        features = torch.tensor(features)
        spectrograms = self.reformer(features).detach()
        return spectrograms

    def generate_performance_audio(self, spectrograms):
        capacity = 80*80*8 #maximum volume of a tensor the wavenet can handle at a time
        hop = floor(capacity / (spectrograms.shape[0] * spectrograms.shape[1]))
        pdb.set_trace()

    def perform(self):
        features = self.generate_performance_features()
        spectrograms = self.generate_performance_spectrograms(features)
        audio = self.generate_performance_audio(spectrograms)
        #have the reformer generate the spectrograms from the features lists






if __name__ == '__main__':
    demo_manager()



