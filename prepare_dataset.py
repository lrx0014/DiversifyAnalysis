import argparse
import os
import tarfile
import zipfile
import requests

from datasets.emg import prepare_emg
from datasets.speech_command import prepare_speech_command

dataset_param_help = "Dataset can be one of: Emg, SpeechCommand"


def preprocess_dataset(dataset_name, data_dir):
    """
    Perform any necessary preprocessing on the dataset.
    """
    print(f"Preprocessing {dataset_name}...")

    if dataset_name == 'EMG':
        prepare_emg(data_dir)
    elif dataset_name == 'SpeechCommand':
        prepare_speech_command(data_dir)
    else:
        print(f"No preprocessing function defined for {dataset_name}\n")
        print(dataset_param_help)


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess dataset.")
    parser.add_argument('--data_dir', required=True, type=str, help="Path to save the dataset.")
    parser.add_argument('--dataset', required=True, type=str, help=dataset_param_help)

    args = parser.parse_args()
    data_dir = args.data_dir
    dataset_name = args.dataset

    # Create data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Download, extract, and preprocess dataset
    preprocess_dataset(dataset_name, data_dir)


if __name__ == "__main__":
    main()
