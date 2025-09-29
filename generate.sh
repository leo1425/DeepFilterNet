export PYTHONPATH=$PWD/DeepFilterNet
echo " -- Setting up environment... -- "
if ! command -v rustup &> /dev/null; then
  echo "Rust is not installed. Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source $HOME/.cargo/env
  echo "Rust installation complete."
else
  echo "Rust is already installed. Skipping installation."
fi
if ! command -v pip3 &> /dev/null; then
  echo "pip3 is not installed."
  sudo apt install python3-pip
  exit 1
else
  echo "pip3 is installed: $(pip3 --version)"
fi

pip3 install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install deepfilternet maturin poetry h5py librosa soundfile
cd ..
mkdir dataset
cd dataset


#!/bin/bash
if [ -z "$(ls -A .)" ]; then
  echo " -- Downloading Valentini dataset... -- "
  curl -L -o valentini-noisy.zip https://www.kaggle.com/api/v1/datasets/download/muhmagdy/valentini-noisy

  sudo apt-get install unzip
  unzip -q valentini-noisy.zip
  rm valentini-noisy.zip
  echo " -- Download complete -- "
else
  echo " -- Skipping download: folder is not empty -- "
fi

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
