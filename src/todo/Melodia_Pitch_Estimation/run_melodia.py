##### Installation instructions #####

### Go to /usr/local/lib/
### Create folder vamp/
### Go to /usr/local/lib/vamp/
### Put the files packaged with this python file into /usr/local/lib/vamp
### They should be:
### /usr/local/lib/vamp/mtg-melodia.cat
### /usr/local/lib/vamp/mtg-melodia.n3
### /usr/local/lib/vamp/mtg-melodia.so


import soundfile
import resampy
import vamp
import argparse
import os
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
import jams


def midi_to_notes(midi, fs, hop, smooth, minduration):

    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    # print(len(midi),len(midi_filt))

    notes = []
    p_prev = -1
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes


def hz2midi(hz):

    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def audio_to_midi_melodia(infile, smooth=0.25, minduration=0.1):

    # define analysis parameters
    fs = 44100
    hop = 128

    # load audio using librosa
    print("Loading audio...")
    data, sr = soundfile.read(infile)
    # mixdown to mono if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    # resample to 44100 if needed
    if sr != fs:
        data = resampy.resample(data, sr, fs)
        sr = fs

    # extract melody using melodia vamp plugin
    print("Extracting melody f0 with MELODIA...")
    melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                          parameters={"voicing": 0.2})

    # hop = melody['vector'][0]
    pitch = melody['vector'][1]

    # impute missing 0's to compensate for starting timestamp
    pitch = np.insert(pitch, 0, [0]*8)

    # debug
    # np.asarray(pitch).dump('f0.npy')
    # print(len(pitch))
    # print(len(list(pitch)))

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(pitch)
   

    # segment sequence into individual midi notes
    notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)
    
    num_to_letter_map = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    for note in notes:
        start_time = note[0]
        duration = note[1]
        note_num = note[2]
        octave = int(note_num / 12) - 1
        letter = num_to_letter_map[note_num % 12]
        print(str(letter) + str(octave), note_num, start_time, duration)

audio_to_midi_melodia("./m4_long_straight_a.wav", 0.25, 0.1)