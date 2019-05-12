import numpy as np
import torch
from torch.utils.data import Dataset
from pure_waves import * #generate_random_sequence, play, pitch, waves, get_bitwave
from utils import mu_law_encode

class SimpleWaveLoader(Dataset):
    """Dataloader for simple waves"""
    def __init__(self, length=100, duration=1, FS=16000, mu_quantization=256):
        self.length = length # "number" of examples. technically doesn't mean much because each example is random anyways
        self.duration = duration
        self.FS = FS
        self.mu_quantization = mu_quantization

    def __len__(self):
        """This dataloader doesn't have a real length because samples are generated randomly on the fly"""
        return self.length

    def __getitem__(self, idx):
        """generate a random sequence from a single wave type"""

        form = waves[np.random.randint(low=0, high=len(waves))]
        wave, conditions = generate_random_sequence(form=form, duration=self.duration, FS=self.FS)
        wave = self.quantize(wave)
        conditions = self.normalize(conditions)
        return conditions, wave

    def normalize(self, conditions):
        """return the normalized form of the conditions"""
        conditions[0] /= pitch('C9') #set the highest reasonable note to a value of 1.0
        # conditions[0] = (conditions[0] - pitch('C4')) / (pitch('C5') - pitch('C4'))
        return torch.tensor(conditions)

    def quantize(self, wave):
        """convert the wave to a discrete integer format"""
        return mu_law_encode(torch.tensor(wave), self.mu_quantization)

    def decode(self, inference):
        """convert the network inference into a sound file"""
        #mu_law decode
        #convert to wave file?
        pass


if __name__ == '__main__':
    #generate a random sequence and play it
    apple, _ = generate_random_sequence(form='triangle', duration=1, FS=16000)
    play(apple, 16000)