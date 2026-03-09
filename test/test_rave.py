import os
import sys
import torch
import torchaudio
import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis

OUTPUT_DIR = "./result/rave_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_audio(tensor, fs, filename):
    tensor = torch.clamp(tensor, -1.0, 1.0)

    tensor_int16 = (tensor * 32767.0).to(torch.int16)

    signal_np = tensor_int16.detach().cpu().numpy()

    signal_np = np.squeeze(signal_np)

    if signal_np.ndim == 2:
        signal_np = signal_np.T

    lps_sig.save_wav(signal_np, fs, filename)


def main():

    if len(sys.argv) < 2:
        print("Uso: python test_rave_compare.py arquivo.wav")
        sys.exit(1)

    wav_path = sys.argv[1]

    waveform, fs = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform / waveform.abs().max()
    waveform = waveform.unsqueeze(0)

    print("Input shape:", waveform.shape)

    pre_trained_model = torch.jit.load("/data/models/vintage.ts")
    pre_trained_model.eval()

    my_model = torch.jit.load("/data/models/first_44k.ts")
    my_model.eval()

    with torch.inference_mode():
        pre_recon = pre_trained_model(waveform)
        my_recon = my_model(waveform)


    in_data = waveform.detach().cpu().squeeze()
    pre_data = pre_recon.detach().cpu().squeeze()
    my_data = my_recon.detach().cpu().squeeze()

    pre_data = pre_data[0, :]

    in_filename = os.path.join(OUTPUT_DIR, "in.wav")
    pre_filename = os.path.join(OUTPUT_DIR, "pre.wav")
    my_filename = os.path.join(OUTPUT_DIR, "my.wav")

    save_audio(in_data, fs, in_filename)
    save_audio(pre_data, fs, pre_filename)
    save_audio(my_data, fs, my_filename)

    psd_filename = os.path.join(OUTPUT_DIR, "psd.png")
    demon_filename = os.path.join(OUTPUT_DIR, "demon.png")
    lofar_filename = os.path.join(OUTPUT_DIR, "lofar.png")

    in_data = in_data.numpy()
    pre_data = pre_data.numpy()
    my_data = my_data.numpy()

    lps_bb.plot_psds(
        filename=psd_filename,
        noises=[in_data, pre_data, my_data],
        labels=["Input", "Pretrained Model", "My Model"],
        fs=lps_qty.Frequency.hz(fs),
        window_size=1024*16,
        overlap=0.5,
    )

    lps_bb.plot_demon_lines(
        filename=demon_filename,
        signals=[in_data, pre_data, my_data],
        labels=["Input", "Pretrained Model", "My Model"],
        fs=lps_qty.Frequency.hz(fs),
    )

    lps_analysis.plot_spectral_analysis(
        filename=lofar_filename,
        signals=[in_data, pre_data, my_data],
        labels=["Input", "Pretrained Model", "My Model"],
        fs=lps_qty.Frequency.hz(fs),
    )


if __name__ == "__main__":
    main()