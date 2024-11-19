import os
import tarfile
import zipfile

import requests


def prepare_emg(data_dir):
    """
    Preprocess dataset EMG.
    """

    # download the dataset
    url = 'https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip'
    dataset_path = os.path.join(data_dir, f"emg.zip")
    print(f"Downloading EMG...")

    response = requests.get(url, stream=True)
    with open(dataset_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"Downloaded EMG to {dataset_path}")

    # extract data
    print(f"Extracting EMG...")
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

    print("Processing dataset EMG done.")