import pkgutil
import requests
import os
from tqdm import tqdm
from zipfile import ZipFile

test_only_celeba_zip = 'https://users.renyi.hu/~gergopool/lucid/test_only_celeba.zip'

def load_layer_info():
    text = pkgutil.get_data(__name__, 'res/layer_info.txt').decode('utf-8')
    data = []
    lines = text.split('\n')
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data

def download_test_data(target_folder):

    root = os.path.join(target_folder, 'test_only_celeba')
    csv = os.path.join(root, 'test.csv')

    if os.path.isdir(root):
        print('Dataset found.')
        return csv
    else:
        print('Dataset not found. Downloading..')

    os.makedirs(target_folder, exist_ok=True)
    zip_path = _download_file(test_only_celeba_zip)
    with ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(target_folder)
    os.remove(zip_path)

    return csv

def _download_file(url):
    local_filename = url.split('/')[-1]
    filepath = os.path.join('/tmp', local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        length = int(r.headers["Content-Length"])
        pbar = tqdm(total=length)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
                pbar.update(8192)
        pbar.close()
    return filepath