#!/bin/bash
set -e  # Exit on error

### COLORS ###
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

### HELPERS ###
log_info()    { echo -e "${YELLOW}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
section()     { echo -e "\n${GREEN}=== $1 ===${NC}\n"; }

### SETUP ENVIRONMENT ###
setup_env() {
  section "Setting up environment"
  export PYTHONPATH=$PWD/DeepFilterNet
  sudo apt-get update && sudo apt-get upgrade -y
  if ! command -v rustup &> /dev/null; then
    log_info "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    log_success "Rust installed: $(rustup --version)"
  else
    log_success "Rust already installed: $(rustup --version)"
  fi

  if ! dpkg -s libhdf5-dev &> /dev/null; then
    log_info "libhdf5-dev not found. Installing..."
    sudo apt-get update && sudo apt-get install -y libhdf5-dev
    log_success "libhdf5-dev installed"
  else
    log_success "libhdf5-dev available"
  fi

  if ! command -v pip3 &> /dev/null; then
    log_info "pip3 not found. Installing..."
    sudo apt-get update && apt-get install -y python3-pip
    log_success "pip3 installed: $(pip3 --version)"
  else
    log_success "pip3 available: $(pip3 --version)"
  fi

  # check if conda is installed
  if ! command -v conda &> /dev/null; then
    log_info "Conda not found. Installing Miniconda..."
    curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
    bash Anaconda3-2025.06-0-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Anaconda3-2025.06-0-Linux-x86_64.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
    conda config --set show_channel_urls yes
    conda config --set auto_activate_base false
    export CONDA_ACCEPT_ALL=true
    log_success "Conda installed: $(conda --version)"
  else
    export CONDA_ACCEPT_ALL=true
    log_success "Conda available: $(conda --version)"
  fi

  # check if conda environment "deepfilternet" exists
  if conda info --envs | grep -q "deepfilternet"; then
    log_success "Conda environment 'deepfilternet' already exists"
  else
    log_info "Creating conda environment 'deepfilternet'..."
    conda create -n deepfilternet python=3.12 -y
    log_success "Conda environment 'deepfilternet' created"
  fi


  log_info "Installing Python dependencies..."
  pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu129
  pip3 install deepfilternet maturin poetry h5py librosa soundfile tqdm icecream
  maturin develop --release -m pyDF-data/Cargo.toml
  log_success "Python dependencies installed"
}

### DOWNLOAD DATASET ###
download_dataset() {
  section "Preparing dataset folder"
  cd ..
  mkdir -p dataset
  cd dataset

  if [ -z "$(ls -A .)" ]; then
    log_info "Dataset folder is empty. Downloading Valentini dataset..."
    curl -L -o valentini-noisy.zip https://www.kaggle.com/api/v1/datasets/download/muhmagdy/valentini-noisy
    sudo apt-get install -y unzip
    unzip -q valentini-noisy.zip
    rm valentini-noisy.zip
    log_success "Dataset downloaded and extracted"
  else
    log_info "Dataset folder not empty. Skipping download"
  fi
}

### GENERATE NOISE FILES ###
generate_noise() {
  section "Generating noise files"
  cd ../DeepFilterNet
  python3 extract_noise.py  ../dataset/clean_testset_wav ../dataset/noisy_testset_wav ../dataset/noise_testset_wav
  python3 extract_noise.py  ../dataset/clean_trainset_56spk_wav ../dataset/noisy_trainset_56spk_wav ../dataset/noise_trainset_56spk_wav
  log_success "Noise files generated"
}

### GENERATE TXT FILES ###
generate_txt() {
  section "Generating TXT files"
  ./generate_text.sh ../dataset/clean_testset_wav/ ../dataset/clean_testset_wav.txt clean_testset_wav
  ./generate_text.sh ../dataset/noise_testset_wav/ ../dataset/noise_testset_wav.txt noise_testset_wav
  ./generate_text.sh ../dataset/clean_trainset_56spk_wav/ ../dataset/clean_trainset_56spk_wav.txt clean_trainset_56spk_wav
  ./generate_text.sh ../dataset/noise_trainset_56spk_wav/ ../dataset/noise_trainset_56spk_wav.txt noise_trainset_56spk_wav
  log_success "TXT files generated"
}

### GENERATE HDF5 FILES ###
generate_hdf5() {
  section "Generating HDF5 files"
  cd ../dataset/
  python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py noise  noise_testset_wav.txt          noise_testset_wav.hdf5
  python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py speech clean_testset_wav.txt          clean_testset_wav.hdf5
  python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py speech clean_trainset_56spk_wav.txt   clean_trainset_56spk.hdf5
  python3 ../DeepFilterNet/DeepFilterNet/df/scripts/prepare_data.py noise  noise_trainset_56spk_wav.txt   noise_trainset_56spk.hdf5
  log_success "HDF5 files generated"
}

### COPY CONFIG ###
copy_cfg() {
  section "Copying dataset.cfg"
  cp ../DeepFilterNet/dataset.cfg .
  log_success "dataset.cfg copied"
}

### MAIN ###
setup_env
download_dataset
generate_noise
generate_txt
generate_hdf5
copy_cfg

section "All done 🎉"
