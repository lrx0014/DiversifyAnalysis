import os
import tarfile
import zipfile

import requests
import urllib

import utils.progress_bar as pbar


def prepare_emg(data_dir):
    """
    Preprocess dataset EMG.
    """

    # download the dataset
    url = 'https://www.dropbox.com/scl/fi/2msk3ke7ly7j6elbb6nn1/emg_processed.zip?rlkey=9a6v1mblczufvz1r0dulob8r7&st=vk36hkdj&dl=1'
    dataset_path = os.path.join(data_dir, f"emg.zip")
    print(f"Downloading EMG...")

    if os.path.exists(dataset_path):
        return
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    urllib.request.urlretrieve(url, dataset_path, pbar.show_progress)
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