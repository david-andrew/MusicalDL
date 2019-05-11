# MusicalDL Final Project



# Requirements
(recommended to use Anaconda to install most of these)
- numpy
- pytorch
- nv-wavenet
- music21	(pip install)
- simpleaudio (pip install)

...





# Notes
Issue with inability to use persistence:
https://github.com/NVIDIA/nv-wavenet/issues/40


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




make sure to use torch.cuda.empty_cache() at the end of each epoch