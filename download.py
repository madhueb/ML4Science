import urllib.request
import argparse
from tqdm import tqdm
import zipfile
import os

""" FILE TO DOWNLOAD THE MIL BENCHMARK DATASETS AND THE CAMELYON16 DATASET """

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
def unzip_data(zip_path, data_path):
    os.makedirs(data_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
        
        
def main():
    """
    Script to download the MIL benchmark datasets and the Camelyon16 dataset into the datasets folder
    Args:
        dataset: str: Dataset to download, default is mil (choices: mil, c16)

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mil', help='Dataset to be downloaded: mil, tcga', choices=['mil', 'c16'])
    args = parser.parse_args()
    
    if args.dataset == "mil":
        print('downloading MIL benchmark datasets')
        download_url('https://uwmadison.box.com/shared/static/arvv7f1k8c2m8e2hugqltxgt9zbbpbh2.zip', 'mil-dataset.zip')
        unzip_data('mil-dataset.zip', 'datasets')
        os.remove('mil-dataset.zip')
    if args.dataset == "c16":
        print('downloading Camelyon16 datasets (pre-computed features)')
        download_url('https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip', 'c16-dataset.zip')
        unzip_data('c16-dataset.zip', 'datasets/Camelyon16')
        os.remove('c16-dataset.zip')
    
if __name__ == '__main__':
    main()