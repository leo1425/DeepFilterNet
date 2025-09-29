export PYTHONPATH=$PWD/DeepFilterNet
echo " -- Setting up environment... -- "
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install deepfilternet
pip install maturin poetry
cd ..
mkdir dataset
cd dataset

echo " -- Downloading Valentini dataset... -- "
#!/bin/bash
curl -L -o valentini-noisy.zip https://www.kaggle.com/api/v1/datasets/download/muhmagdy/valentini-noisy

sudo apt-get install unzip
unzip valentini-noisy.zip
rm valentini-noisy.zip

cd ../DeepFilterNet

echo " -- Generate noise files -- "
python3 extract_noise.py  ../dataset/clean_testset_wav ../dataset/noisy_testset_wav ../dataset/noise_testset_wav
python3 extract_noise.py  ../dataset/clean_trainset_56spk_wav ../dataset/noisy_trainset_56spk_wav ../dataset/noise_trainset_56spk_wav
echo " -- Noise files generated -- "


echo " -- Generate TXT files --"
./generate_text.sh ../dataset/clean_testset_wav/ ../dataset/clean_testset_wav.txt clean_testset_wav
./generate_text.sh ../dataset/noise_testset_wav/ ../dataset/noise_testset_wav.txt noise_testset_wav
./generate_text.sh ../dataset/clean_trainset_56spk_wav/ ../dataset/clean_trainset_56spk_wav.txt clean_trainset_56spk_wav
./generate_text.sh ../dataset/noise_trainset_56spk_wav/ ../dataset/noise_trainset_56spk_wav.txt noise_trainset_56spk_wav
echo " -- TXT files generated -- "

echo " -- Generate HDF5 files -- "
cd ../dataset/
python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py noise noise_testset_wav.txt noise_testset_wav.hdf5
python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py speech clean_testset_wav.txt clean_testset_wav.hdf5
python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py speech clean_trainset_56spk_wav.txt clean_trainset_56spk.hdf5
python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py noise noise_trainset_56spk_wav.txt noise_trainset_56spk.hdf5
echo " -- HDF5 files generated -- "

echo" -- Copy dataset.cfg -- "
cp ../DeepFilterNet/dataset.cfg .
echo " -- dataset.cfg copied -- "

echo " -- All done -- "
