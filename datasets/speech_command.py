import os
import pathlib
import sys
import urllib.request
import tarfile
import zipfile
import requests
import torch
import torchaudio
import sklearn.model_selection
import controldiffeq

import utils.progress_bar as pbar

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))


def prepare_speech_command(data_dir):
    """
    Preprocess dataset Speech-Command.
    """

    get_data('', 200, data_dir)

    print("Processing dataset Speech-Command done.")


def download(data_dir):
    base_loc = data_dir
    loc = os.path.join(base_loc, 'speech_commands.tar.gz')
    # loc = base_loc / 'speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    print("downloading dataset file...")
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc, pbar.show_progress)
    with tarfile.open(loc, 'r') as f:
        f.extractall(base_loc)
    print(f"downloaded dataset file to {base_loc}.")


def _process_data(loc, intensity_data, data_dir):
    print("processing dataset file...")
    base_loc = data_dir
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = os.path.join(base_loc, foldername)
        # loc = base_loc / foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load(os.path.join(loc, filename), channels_first=False,
                                           normalize=False)  # for forward compatbility if they fix it
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                   melkwargs=dict(n_fft=200, n_mels=64))(X.squeeze(-1)).transpose(1, 2).detach()
    # X is of shape (batch=34975, length=161, channels=20)

    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = preprocess_data(loc, times, X, y, final_index, append_times=True,
                                                   append_intensity=intensity_data)

    print(f"processed dataset files to {data_dir}.")

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(intensity_data, batch_size, data_dir):
    base_base_loc = os.path.join(data_dir, f"processed_data")
    loc = os.path.join(base_base_loc, ('speech_commands_with_mels' + ('_intensity' if intensity_data else '')))
    # loc = base_base_loc / ('speech_commands_with_mels' + ('_intensity' if intensity_data else ''))
    if os.path.exists(loc):
        print("dataset path exist, processing...")
        tensors = load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        download(data_dir)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(loc, intensity_data, data_dir)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader


def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data(loc, times, X, y, final_index, append_times, append_intensity):
    X = normalise_data(X, y)

    # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
    # depend on that order.
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if append_intensity:
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype).cumsum(dim=1)
        augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)

    train_X, val_X, test_X = split_data(X, y)
    train_y, val_y, test_y = split_data(y, y)
    train_final_index, val_final_index, test_final_index = split_data(final_index, y)

    save_data(loc, _train_X=train_X, _val_X=val_X,_test_X=test_X, _train_y=train_y,_val_y=val_y,_test_y=test_y)

    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X)
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X)

    in_channels = X.size(-1)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)


def wrap_data(times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
              test_final_index, device, batch_size, num_workers=4):
    times = times.to(device)
    train_coeffs = tuple(coeff.to(device) for coeff in train_coeffs)
    val_coeffs = tuple(coeff.to(device) for coeff in val_coeffs)
    test_coeffs = tuple(coeff.to(device) for coeff in test_coeffs)
    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_final_index = train_final_index.to(device)
    val_final_index = val_final_index.to(device)
    test_final_index = test_final_index.to(device)

    train_dataset = torch.utils.data.TensorDataset(*train_coeffs, train_y, train_final_index)
    val_dataset = torch.utils.data.TensorDataset(*val_coeffs, val_y, val_final_index)
    test_dataset = torch.utils.data.TensorDataset(*test_coeffs, test_y, test_final_index)

    train_dataloader = dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return times, train_dataloader, val_dataloader, test_dataloader


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(os.path.join(dir, tensor_name)) + '.pt')


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def download_sc_directly(data_dir):
    """
    download preprocessed speech-command dataset directly from cloud drive.
    """

    # download the dataset
    url = 'https://www.dropbox.com/scl/fi/q5t42y4w753b94ked3y61/SpeechCommand_24482.zip?rlkey=ely4avzkhitivtymz4sjnwd23&st=boqnsaov&dl=1'
    dataset_path = os.path.join(data_dir, f"sc_processed_data.zip")
    
    print(f"[dropbox] Downloading SpeechCommand...")

    if os.path.exists(dataset_path):
        return
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    urllib.request.urlretrieve(url, dataset_path, pbar.show_progress)
    print(f"Downloaded SpeechCommand to {dataset_path}")

    # extract data
    print(f"Extracting SpeechCommand...")
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

    print("Processing dataset SpeechCommand done.")