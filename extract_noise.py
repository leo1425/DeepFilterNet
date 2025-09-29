import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm  # progress bar

def extract_noise(clean_file, noisy_file, output_file):
    # Load both files
    clean, sr_clean = librosa.load(clean_file, sr=None)
    noisy, sr_noisy = librosa.load(noisy_file, sr=None)

    if sr_clean != sr_noisy:
        raise ValueError(f"Sample rates do not match for {clean_file} and {noisy_file}")

    # Ensure same length
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]

    # Noise = noisy - clean
    noise = noisy - clean

    # Normalize to avoid clipping
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))

    # Save result
    sf.write(output_file, noise, sr_clean)


def main(clean_dir, noisy_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    clean_files = {f: os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".wav")}
    noisy_files = {f: os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".wav")}

    common_files = set(clean_files.keys()) & set(noisy_files.keys())

    if not common_files:
        print("No matching files found between clean and noisy folders.")
        return

    # tqdm progress bar
    for fname in tqdm(sorted(common_files), desc="Extracting noise", unit="file"):
        clean_path = clean_files[fname]
        noisy_path = noisy_files[fname]
        output_path = os.path.join(output_dir, fname)

        try:
            extract_noise(clean_path, noisy_path, output_path)
        except Exception as e:
            tqdm.write(f"Failed on {fname}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract noise from paired clean and noisy WAV files.")
    parser.add_argument("clean_dir", help="Folder containing clean speech WAV files")
    parser.add_argument("noisy_dir", help="Folder containing noisy speech WAV files")
    parser.add_argument("output_dir", help="Folder to save extracted noise WAV files")

    args = parser.parse_args()
    main(args.clean_dir, args.noisy_dir, args.output_dir)
