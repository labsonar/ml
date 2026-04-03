import os
import argparse
import glob
import numpy as np
import torch
import torchaudio

import scipy.io.wavfile as scipy_wav

import lps_ml.utils.sonar_loss as sonar_loss
import lps_ml.audio_processors.time_processors as time_processors

def load_wav(path):
    data, sr = torchaudio.load(path)  # [C, T]
    return sr, data


def apply_to_float(converter, sr, data):
    data_np = data.numpy()
    sr_out, data_float = converter.process(sr, data_np)
    tensor = torch.from_numpy(data_float).unsqueeze(0)  # [1, C, T] ou [1, T]
    return tensor


def get_wav_files(input_paths):
    files = []

    for p in input_paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.wav")))
        elif os.path.isfile(p) and p.endswith(".wav"):
            files.append(p)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Compute SonarLoss for WAV files")

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Lista de arquivos .wav ou diretórios"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/sonar_loss_plots",
        help="Diretório de saída"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    converter = time_processors.ToFloatConverter()
    loss = sonar_loss.SonarLoss()

    wav_files = get_wav_files(args.inputs)

    if len(wav_files) < 1:
        raise RuntimeError("Nenhum arquivo WAV encontrado")

    print(f"Encontrados {len(wav_files)} arquivos")

    inputs = []
    names = []

    for f in wav_files:

        fs, data = scipy_wav.read(f)
        fs, data = converter.process(fs=fs, data=data)

        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0).unsqueeze(0)

        print("data: ", data.shape)
        inputs.append(data)
        names.append(os.path.basename(f))

    loss.plot(inputs=inputs, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
