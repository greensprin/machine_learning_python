# coding: utf-8

#download mnist
import urllib.request
import os.path
#data load
import numpy as np
import gzip
from PIL import Image

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
  'train_img':'train-images-idx3-ubyte.gz',
  'train_label':'train-labels-idx1-ubyte.gz',
  'test_img':'t10k-images-idx3-ubyte.gz',
  'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))

#data offset
img_offset = 16
label_offset = 8

def _download(filename):
    file_path = dataset_dir + '/' + filename
    if os.path.exists(file_path):
        return print('already exist')
    print('Downloading ' + filename + ' ...')
    urllib.request.urlretrieve(url_base + filename, file_path)
    print('Done')

def download_mnist():
    for v in key_file.values():
       _download(v)

#ファイルを指定して取得
def load_mnist(filename):
  file_path = dataset_dir + '/' + filename
  with gzip.open(file_path, 'rb') as f:
# file read
    content = f.read()
#read img cnt
    img_cnt = int.from_bytes(content[ 4: 8], 'big')
    if (filename == key_file['train_img'] or filename == key_file['test_img']):
       d_offset = img_offset
    elif (filename == key_file['train_label'] or filename == key_file['test_label']):
       d_offset = label_offset
    else:
      return print("no such file")
    data = np.frombuffer(content, np.uint8, offset=d_offset)
  return data.reshape(-1, img_cnt)

#すべてのデータを一括で取得
def all_data_load():
  trn_data = load_mnist(key_file['train_img'])
  trn_lbl = load_mnist(key_file['train_label'])
  tst_data = load_mnist(key_file['test_img'])
  tst_lbl = load_mnist(key_file['test_label'])
  return (trn_data, trn_lbl, tst_data, tst_lbl)

if __name__ == '__main__':
  download_mnist()

  data = load_mnist(key_file['train_img'])
  print(data[0].shape)
  # img1 = data[0].reshape(28, 28)
  # 画像表示
  # pil_img = Image.fromarray(np.uint8(img1))
  # pil_img.show()


