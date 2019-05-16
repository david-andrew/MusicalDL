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
import soundfile
import subprocess

#autotuning
import librosa
from librosa.effects import pitch_shift

#ML models
import transformer
import nv_wavenet
import utils

#simple file select dialog window
import tkinter
from tkinter import filedialog

#data observation
from matplotlib import pyplot as plt
from pure_waves import play


class demo_manager():
    def __init__(self, section_size=1):
        self.FS = 16000
        self.SS = 80
        assert((self.FS / self.SS) % 1 == 0) #ensure that spectrogram scale is an even multiple of sample rate
        self.sample_conversion = int(self.FS / self.SS)#conversion from SS to FS

        self.section_size = section_size #number of singers per each section size

        #list which singer indices map to which voice types
        self.singers =  {
            'Soprano':  [0],#[0, 1, 2, 3, 4, 5, 6, 7, 8],
            'Alto':     [0],#[0, 1, 2, 3, 4, 5, 6, 7, 8],
            'Tenor':    [10],#[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Baritone': [10],#[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Bass':     [10],#[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
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
        self.save_path = os.path.join('.', 'output')
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
        print('Synthesizing performance of %s' % self.song_name)
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

        self.spec_length = ceil(self.SS * self.duration)
        self.audio_length = self.spec_length * self.sample_conversion #total number of samples


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

        part_features = np.zeros((27, self.spec_length), dtype=np.float32) #27 channels for the feature data
        part_features[2+singer_idx] = 1 #set selcted singer for feature vector

        start = 0
        for note in part:
            stop = start + ceil(note['duration'] * self.SS)        
            if note['volume'] != 0:
                part_features[0, start:stop] = note['pitch'] / pitch('C8')
                part_features[1, start:stop] = note['volume'] / 3
                part_features[22+0] = 1 #set the vowel spoken to Ah (probably change later)
            #else leave as all zeros
            start = stop #update current index in part

        return part_features

    def generate_performance_spectrograms(self, features):
        features = np.stack(features, 0)
        features = torch.tensor(features)
        spectrograms = self.reformer(features).detach()
        return spectrograms

    def generate_performance_audio(self, spectrograms):
        batch_size = spectrograms.shape[0] #number of voices
        spec_depth = spectrograms.shape[1] #depth of each spectrogram. should be 80

        capacity = 80*80*9 #maximum volume of a tensor the wavenet can handle at a time
        spec_hop = floor(capacity / (batch_size * spec_depth))
        spec_start = 0
        audio_start = 0
        audio_hop = spec_hop * self.sample_conversion
        batch = 1
        num_batches = ceil(self.spec_length / spec_hop)

        spectrograms = utils.to_gpu(spectrograms)
        audio = np.zeros((batch_size, self.audio_length))

        print('Generating audio with WaveNet Vocoder...')
        while spec_start + spec_hop < self.spec_length:
            print(' - batch %d of %d' % (batch, num_batches))
            #get clip
            clip = spectrograms[:, :, spec_start:spec_start+spec_hop]

            #get audio from network
            cond_input = self.wavenet.get_cond_input(clip)
            audio_data = self.nv_wavenet.infer(cond_input, nv_wavenet.Impl.AUTO)
            torch.cuda.empty_cache()
            # pdb.set_trace()
            for i in range(batch_size):
                audio[i, audio_start:audio_start+audio_hop] = utils.mu_law_decode_numpy(audio_data[i,:].cpu().numpy(), self.nv_wavenet.A)

            #add into at start:start+hop
            spec_start += spec_hop
            audio_start += audio_hop
            batch += 1

            #need to update the wavenet embeddings so that sound stream is continuous
            #here there be demons
            # self.nv_wavenet.embedding_prev = self.nv_wavenet.embedding_curr

        #add the last section if it didn't fit
        print(' - batch %d of %d' % (batch, num_batches))
        # spec_remaining = self.spec_length - spec_start
        clip = spectrograms[:, :, spec_start:self.spec_length]

        #get audio from network
        cond_input = self.wavenet.get_cond_input(clip)
        audio_data = self.nv_wavenet.infer(cond_input, nv_wavenet.Impl.AUTO)
        torch.cuda.empty_cache()
        for i in range(batch_size):
            audio[i, audio_start:self.audio_length] = utils.mu_law_decode_numpy(audio_data[i,:].cpu().numpy(), self.nv_wavenet.A)

        return audio        

    def mask_rests(self, audio):
        """because the network doesn't understant silence, need to manually mask rests in singers parts"""
        for part_idx, (singer, part) in enumerate(self.song.items()):
            for section_idx in range(self.section_size):
                i = part_idx * self.section_size + section_idx
                print('Masking %s #%d rests...' % (singer, section_idx))
                start = 0
                for note in part:
                    duration = ceil(note['duration'] * self.FS)
                    if note['volume'] == 0:
                        audio[i, start:start+duration] = 0 #mask all rests in audio
                    start += duration
        return audio


    def semitone_diff(self, actual, desired):
        return (12/np.log(2)) * np.log(desired / actual);

    def autotune(self, audio):
        """because the singers can't sing in tune for their livs"""
        for part_idx, (singer, part) in enumerate(self.song.items()):
            for section_idx in range(self.section_size):
                i = part_idx * self.section_size + section_idx 
                print('Autotuning %s #%d...' % (singer, section_idx))

                #write that audio to disk and pitch detect
                audio_path = os.path.join(self.save_path, 'tmp', 'raw' + self.song_name + '_' + singer + '.wav')
                pitch_path = os.path.join(self.save_path, 'tmp', 'raw' + self.song_name + '_' + singer + '.f0')
                reaper_path = os.path.join('.', 'REAPER', 'build', 'reaper')

                soundfile.write(audio_path, audio[i], self.FS)

                subprocess.check_output([reaper_path, '-i', audio_path, '-f', pitch_path, '-a'])

                with open(pitch_path, 'r') as f:
                    reaper_output = f.read()

                reaper_output = reaper_output.split('\n')[7:-1] #skip header information
                time, voice, f0 = [], [], []
                for line in reaper_output:
                    t, v, f = line.split(' ')
                    time.append(float(t))
                    voice.append(int(v))
                    f0.append(float(f))

                
                # start = 0
                reaper_sample_rate = time[1] - time[0]
                time0 = 0 #current time
                time1 = 0 #
                for note in part:
                    time1 = time0 + note['duration']
                    if note['volume'] != 0:
                        #compute pitch at that interval
                        start = int(time0/reaper_sample_rate)
                        stop = int(time1/reaper_sample_rate)
                        actual_pitch = np.asarray([f for i, f in enumerate(f0[start:stop]) if voice[i] == 1]).mean()
                        desired_pitch = note['pitch']
                        semitone_shift = self.semitone_diff(actual_pitch, desired_pitch)

                        #perform autotuning
                        start = int(time0 * self.FS)
                        stop = int(time1 * self.FS)
                        audio[i, start:stop] = pitch_shift(audio[i, start:stop], self.FS, semitone_shift)




                    time0 = time1

        return audio


    def perform(self):
        features = self.generate_performance_features()
        spectrograms = self.generate_performance_spectrograms(features)
        audio = self.generate_performance_audio(spectrograms)

        #apply silence to all areas of rests in audio
        audio = self.mask_rests(audio)

        #autotune the audio
        # audio = self.autotune(audio)

        #merge all of the voices together
        audio = np.mean(audio, axis=0) #merge all of the voices together
        
        #post process the audio (reverb)

        #save audio


        soundfile.write(os.path.join(self.save_path, self.song_name + '.wav'), audio, self.FS)



if __name__ == '__main__':
    demo_manager()



