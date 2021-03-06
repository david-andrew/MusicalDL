# MusicalDL
EN.601.682 Machine Learning: Deep Learning final project by William David, Sophia Doerr, David Samson. 

## Results and Examples
Network audio results can be found in `./src/output`. The best example is `./src/output/tuned/Danny_Boy.wav`. Poster for poster session can be found at `./misc/presentation/MusicalDL Poster Session Poster.pdf`.

`./src/output/untuned` is the raw network output. Currently the network has very poor pitch comprehension, so all of these examples are grossly out of tune

`./src/output/tuned` is the raw network output with an autotuner applied to correct the pitch. The autotuner was very rough, and so most of the examples don't sound very good. Best sounding example is Danny Boy.

Examples of spectrograms for the transformer step can be found in `./src/models/vocal_transformer`


## Goal
Create a simple but realistic sounding choral synthesizer. For the project the goal is to synthesize a singer with control over the following parameters
- voice 	(e.g. the different singers in vocalset)
- pitch
- volume
- phoneme 	(from a small selection, namely 'ah' 'eh' 'ee' 'oh' 'oo', and perhaps 'mm' 'nn' 'll')




## Approach
Train a wavenet on examples of synthesized and real voices, while conditioning on parameters such as pitch, volume, and phoneme. This will produce a single voice wavenet synthesizer, which can then be layered with multiple instances, and controlled to produce a choral synthesis (control might be provided via sheet music, midi input, or some other musical interface)




## Requirements
(recommended to use Anaconda to install most of these)
- numpy
- pytorch
- cuda
- music21		(pip install)
- simpleaudio 	(pip install)
- librosa 		(only to run nv-wavenet examples)




## Getting started
1. Clone this repository

```
$ git clone https://github.com/david-andrew/MusicalDL
```


2. Set up the nv-wavenet submodule (for more info, see: https://git-scm.com/book/en/v2/Git-Tools-Submodules)

```
$ git submodule init
$ git submodule update
```

Build the nv-wavenet repo (see instructions under "Try It" in https://github.com/NVIDIA/nv-wavenet/tree/master/pytorch)
Note: You will probably need to adjust `src/nv_wavenet/pytorch/Makefile` line 30 `ARCH=sm_70` with your cuda configuration, e.g. for my nvidia GTX 1070 the correct setting is `ARCH=sm_61`. See 

```
$ cd src/nv_wavenet/pytorch
$ make
$ python build.py install
```

Test nv-wavenet with their example model and conditional data to verify the setup
Note: On my setup, in `nv_wavenet_test.py` the line 40 `samples = wavenet.infer(cond_input, nv_wavenet.Impl.PERSISTENT)[0]` needed to be changed to `samples = wavenet.infer(cond_input, nv_wavenet.Impl.AUTO)[0]`
- download `model.pt` and `cond_input.pt` from the "Try It" section and place them in `src/nv_wavenet/pytorch`

```
$ python nv_wavenet_test.py 
```

This should produce `audio.wav` which was generated using the wavenet


3. Run the music synthesizer
TODO: implement the synthesizer and write this section. Will probably have a command-line interface as well as an interface that uses https://github.com/david-andrew/Ensemble in order to perform the sheet music to synthesizer stream conversion



## Notes and Issues:
- for my system, all instances of `nv_wavenet.Impl.PERSISTENT` needed to be converted to `nv_wavenet.Impl.AUTO`
- due to updates in pytorch, in `train.py` on line 143  `loss.data[0]` needs to be modified to `loss.data.item()`
- make sure to use `torch.cuda.empty_cache()` at the end of each epoch
- Issue with inability to use persistence: https://github.com/NVIDIA/nv-wavenet/issues/40




## TODO
Set up proper submodule installation for calling wavenet from the project. At the moment, its probably not the best idea to simply add the wavenet path to sys.path? Look into this later
Set up my cuda install to be able to use persistent if possible




## misc notes to clean up later
Notes for demonstrating proficiency in the project:
- why you select design choices
- how you approach solving the problem?
- why something fails
- changes you make to fix things that fail

Goal:
- create a choral voice synthesizer
- be able to synthesize my voice
- be able to create music that sounds good with synthesized voices

Problems:
- I can't record enough of my voice -> train network with lots of other voices first then train with mine to teach my voice and transfer learned knowledge to generate my voice
- vocalset doesn't have enough data -> lots of data augmentation: splice, pitch-shift, amplitude-shift?