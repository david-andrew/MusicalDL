cd nv_wavenet/pytorch
ls ../../data/VocalSet/training_clips/*.wav > test_files.txt
python mel2samp_onehot.py -a test_files.txt -o ../../data/VocalSet/training_clips/ -c config.json
cd ../..