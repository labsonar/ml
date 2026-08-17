#!/usr/bin/env python3

import os
import argparse

import soundfile as sf

import lps_utils.utils as lps_utils

def split_wav_file(
    input_file: str,
    output_dir: str,
    fragment_duration: float,
    max_frags: int
) -> None:
    """
    Divide um arquivo WAV em fragmentos de duração fixa.
    """

    data, fs = sf.read(input_file)

    samples_per_fragment = int(fragment_duration * fs)

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    n_fragments = (len(data) + samples_per_fragment - 1) // samples_per_fragment

    for i in range(n_fragments):

        if i >= max_frags:
            break

        start = i * samples_per_fragment
        end = min(start + samples_per_fragment, len(data))

        fragment = data[start:end]

        output_file = os.path.join(
            output_dir,
            f"{base_name}_frag{i+1}.wav"
        )

        sf.write(output_file, fragment, fs)


        print(f"Saved: {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Fragmenta arquivos WAV em trechos de duração fixa."
    )

    parser.add_argument(
        "input_dir",
        help="Diretório contendo os arquivos WAV."
    )

    parser.add_argument(
        "output_dir",
        help="Diretório de saída."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Duração dos fragmentos em segundos (default: 15)."
    )

    parser.add_argument(
        "--max_frags",
        type=int,
        default=None,
        help="Duração dos fragmentos em segundos (default: 15)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wav_files = lps_utils.find_files(args.input_dir)

    for wav_file in wav_files:

        input_path = os.path.join(args.input_dir, wav_file)

        try:
            split_wav_file(
                input_file=input_path,
                output_dir=args.output_dir,
                fragment_duration=args.duration,
                max_frags=args.max_frags
            )

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")


if __name__ == "__main__":
    main()