import __init__
import torch
import nv_wavenet
from torch.utils.data import DataLoader
from loader import SimpleWaveLoader
from pure_waves import play
import utils
import time
import pdb

#load the model
model = torch.load('models/simple/checkpoint_6972')['model']
# model.eval()
# wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))

#create some test data to generate
testset = SimpleWaveLoader()
test_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None, batch_size=8, pin_memory=False, drop_last=True)


for batch in test_loader:
    # conditions, true_audio = testset[0]#batch

    x, y = batch
    true_audio = y.clone()

    y = torch.zeros_like(y) #removing the waveform for pure inference
    x = utils.to_gpu(x).float()
    y = utils.to_gpu(y)
    x = (x, y)  # auto-regressive takes outputs as inputs
    y_pred = model(x)
    single = y_pred[0].detach().cpu()
    values, indices = single.max(0)
    indices = utils.mu_law_decode_numpy(indices.numpy(), 256)
    indices = utils.MAX_WAV_VALUE * indices
    indices = indices.astype('int16')

    true_audio = utils.mu_law_decode_numpy(true_audio[0].cpu().numpy(), 256)
    true_audio = utils.MAX_WAV_VALUE * true_audio
    true_audio = true_audio.astype('int16')

    play(indices, 16000)
    time.sleep(0.25)
    play(true_audio, 16000)
    time.sleep(1.0)

    del x, y, y_pred, single, values, indices, true_audio
    torch.cuda.empty_cache()