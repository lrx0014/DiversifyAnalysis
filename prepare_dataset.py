import argparse
import os
import tarfile
import zipfile

import requests


def download_dataset(dataset_name, data_dir):
    """
    Download the specified dataset to the given directory.
    """
    dataset_urls = {
        'EMG': 'https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip',
        'Speech_Commands': 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
        # Add more datasets here
    }

    if dataset_name not in dataset_urls:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(dataset_urls.keys())}")

    url = dataset_urls[dataset_name]
    dataset_path = os.path.join(data_dir, f"{dataset_name}.zip")
    print(f"Downloading {dataset_name}...")

    response = requests.get(url, stream=True)
    with open(dataset_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {dataset_name} to {dataset_path}")
    return dataset_path


def extract_dataset(dataset_path, data_dir):
    """
    Extract the downloaded dataset.
    """
    print(f"Extracting {dataset_path}...")
    if dataset_path.endswith(".zip"):
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    elif dataset_path.endswith((".tar", ".tar.gz", ".tar.bz2")):
        with tarfile.open(dataset_path, 'r:*') as tar_ref:
            tar_ref.extractall(data_dir)
    else:
        raise ValueError("Unsupported file format. Please provide a zip, tar, tar.gz, or tar.bz2 file.")

    print(f"Extracted to {data_dir}")

    # Clean up the zip file
    os.remove(dataset_path)


def preprocess_dataset(dataset_name, data_dir):
    """
    Perform any necessary preprocessing on the dataset.
    """
    print(f"Preprocessing {dataset_name}...")

    if dataset_name == 'EMG':
        preprocess_emg(data_dir)
    else:
        print(f"No preprocessing function defined for {dataset_name}")


def preprocess_emg(data_dir):
    """
    Preprocess dataset EMG.
    """
    print("Processing dataset EMG done.")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess dataset.")
    parser.add_argument('--data_dir', required=True, type=str, help="Directory to save the dataset.")
    parser.add_argument('--dataset', required=True, type=str, help="Name of the dataset to download and preprocess.")

    args = parser.parse_args()
    data_dir = args.data_dir
    dataset_name = args.dataset

    # Create data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Download, extract, and preprocess dataset
    dataset_path = download_dataset(dataset_name, data_dir)
    extract_dataset(dataset_path, data_dir)
    preprocess_dataset(dataset_name, data_dir)


if __name__ == "__main__":
    main()
