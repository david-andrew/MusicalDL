# import __init__
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

from math import log as ln, ceil
import soundfile
import resampy
import vamp
import json


class VocalSetLoader(Dataset):
    """"""
    def __init__(self, path=os.path.join('.', 'data', 'VocalSet'), length=100, duration=1, FS=16000, mu_quantization=256):
        self.length = length # "number" of examples. technically doesn't mean much because each example is random anyways
        self.duration = duration
        self.FS = FS
        self.mu_quantization = mu_quantization
        self.path = os.path.join(path, 'FULL')
        self.save_path = os.path.join(path, 'training_clips')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.voices = voices = [voice for voice in os.listdir(self.path) if re.match(r'(fe)?male[0-9]+', voice) is not None]
        self.techniques = ['arpeggios', 'long_tones', 'scales']
        self.exclusions = ['trill', 'lip_trill', 'vocal_fry', 'inhaled', 'trillo', 'breathy', '_DS_Store', '.DS_Store'] #maybe 'messa'?
        self.vowels = ['a', 'e', 'i', 'o', 'u']

        self.vec_freq_to_midi = np.vectorize(lambda f: self.freq_to_midi(f) if f > 0 else -1)
        self.hop = None
        self.voice_vowel_combos = []

    def __len__(self):
        return self.length

    def freq_to_midi(self, freq):
        """convert a frequency to it's nearest midi-number"""
        return round((12/ln(2)) * ln(freq/27.5) + 21)

    def get_all_files(self):
        """return a list of the paths to every wave file to be used"""
        files = []
        for voice in self.voices:
            voice_path = os.path.join(self.path, voice)
            for technique in [t for t in os.listdir(voice_path) if t in self.techniques]:
                technique_path = os.path.join(voice_path, technique)
                for quality in [q for q in os.listdir(technique_path) if q not in self.exclusions]:
                    quality_path = os.path.join(technique_path, quality)
                    for filename in [w for w in os.listdir(quality_path) if re.match(r'.*\.wav', w) is not None]:
                        files.append(os.path.join(quality_path, filename))
        return files

    def features_from_filename(self, filepath):
        """return a tuple of the vocalset features based on the filename"""
        filename = os.path.basename(filepath)
        features = os.path.splitext(filename)[0].split('_')

        assert(features[-1] in self.vowels)
        return (features[0], features[-1]) #return only the singer and the vowel

    def load_wav(self, filepath):
        """load the wave file as a numpy array. Adapted from run_melodia.py written by Will"""
        data, sr = soundfile.read(filepath)
        
        # mixdown to mono if needed
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        
        # resample to 16000 if needed
        if sr != self.FS:
            data = resampy.resample(data, sr, self.FS)

        return data

    def get_pitches(self, wave, convert_to_midi=False):
        """return the pitches associated with this wave file. Adapted from run_melodia.py written by Will"""
        
        # extract melody using melodia vamp plugin
        # print("Extracting melody f0 with MELODIA...")
        melody = vamp.collect(wave, self.FS, "mtg-melodia:melodia", parameters={"voicing": 0.2})

        # hop = melody['vector'][0]
        pitch = melody['vector'][1]

        # impute missing values to compensate for starting timestamp
        pitch = np.insert(pitch, 0, [-1]*8)
        self.hop = float(melody['vector'][0])

        #convert Hz pitches to midi pitch numbers
        if convert_to_midi:
            pitch = self.vec_freq_to_midi(pitch)

        return pitch

    def get_sequences(self, midi_pitches):
        """count the lengths of the midi_sequences. Used to determine how long each note is held for"""
        
        length = 0      #length of current sequence
        index = 0       #starting index of current sequence
        pitch = -1      #pitch of current sequence
        sequences = {}  #dictionary to store sequence indices and lengths
        
        #loop over the list of pitches and count the contiguous number of repeated pitches
        while index + length < len(midi_pitches):
            if midi_pitches[index + length] != pitch:
                if pitch >= 0: #only record actual pitches
                    sequences[index] = length
                pitch = midi_pitches[index + length]
                index += length
                length = 0
            else:
                length += 1

        #add the last sequence missed by the list
        if pitch >= 0:
            sequences[index] = length 
        
        #filter only sequences that are longer than desired clip length
        sequences = {k:v for k,v in sequences.items() if v * self.hop > self.duration} 
        
        return sequences

    def extract_clips(self, wave, sequences, pitch):
        """return a list of wave clips of the desired clip length"""
        clips = []
        clip_hops = ceil(self.duration / self.hop) #number of hops in a single clip
        stride = int(clip_hops * 0.5)    #amount to shift window when extracting adjacent clips

        #loop over the list of sequences
        for index, length in sequences.items():
            i = index
            while i + clip_hops < index + length:
                start = int(i * self.hop * self.FS)
                end = int((i + clip_hops) * self.hop * self.FS)
                clips.append((int(pitch[i]), wave[start:end])) #tuple of the clip and its pitch
                i += stride
        return clips

    def get_velocity(self, clip):
        """compute the midi velocity of the clip. roughly convert the clip to 1-127 range using mean absolute value"""
        return int(np.clip(int(ceil(np.absolute(clip).mean() * 500)), 1, 127))

    def save_clips(self, clips, filepath, filenum):
        """save each of the clips in the list along with corresponding jsons"""
        voice, vowel = self.features_from_filename(filepath)
        voice_vowel = '_'.join([voice, vowel])
        if voice_vowel not in self.voice_vowel_combos:
            self.voice_vowel_combos.append(voice_vowel)
        voice_vowel_num = self.voice_vowel_combos.index(voice_vowel)

        for i, (pitch, clip) in enumerate(clips):
            clip_name = '_'.join([str(filenum),voice,vowel,str(pitch),str(i)])
            features = {
                clip_name : {
                    "note": 201034,                                 #unchanged
                    "sample_rate": 16000,                           #unchanged
                    "instrument_family": 0,                         #unchanged
                    "qualities": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],    #unchanged
                    "instrument_source_str": "synthetic",           #unchanged
                    "note_str": "bass_synthetic_033-022-050",       #unchanged
                    "instrument_family_str": "bass",                #unchanged
                    "instrument_str": "bass_synthetic_033",         #unchanged
                    "pitch": pitch,                                    
                    "instrument": 417,                              #unchanged
                    "velocity": self.get_velocity(clip),
                    "instrument_source": voice_vowel_num,
                    "qualities_str": ["dark"]                       #unchanged
                }
            }
            with open(os.path.join(self.save_path, clip_name + '.json'), 'w') as fp:
                json.dump(features, fp)
            soundfile.write(os.path.join(self.save_path, clip_name + '.wav'), clip, self.FS)


    def construct_gansynth_examples(self, clip_length=0.5):
        """construct 0.5-second single pitch examples + corresponding jsons"""
        #walk over all examples
        #get pitch for all examples
        #find sections with 0.5 of contiguous pitch
        #(todo)apply pitch shifts to whole
        #extract sound clips
        #create json

        self.duration = clip_length

        files = self.get_all_files()
        for filenum, filepath in enumerate(files):
            filename = os.path.basename(filepath)
            print('Saving %s clips...' % filename)
            wave = self.load_wav(filepath)
            pitch = self.get_pitches(wave, convert_to_midi=True)
            sequences = self.get_sequences(pitch)
            clips = self.extract_clips(wave, sequences, pitch)
            self.save_clips(clips, filepath, filenum)

            



    def construct_wavenet_examples(self):
        """construct 1-seconds examples and save to the disk for easy loading"""
        # save_path = os.path.join(self.path, 'train')
        # voices = [voice for voice in os.listdir(self.path) if re.match(r'(fe)?male[0-9]+', voice) is not None]
        # clips = ['arpeggios', 'long_tones', 'scales']
        # qualities = None
        # pdb.set_trace()




if __name__ == '__main__':
    v = VocalSetLoader()
    v.construct_gansynth_examples()