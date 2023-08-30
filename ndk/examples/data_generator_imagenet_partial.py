# -*- coding: utf-8 -*-
"""
Copyright(c) 2018 by Ningbo XiTang Information Technologies, Inc and
WuQi Technologies, Inc. ALL RIGHTS RESERVED.

This Information is proprietary to XiTang and WuQi, and MAY NOT be copied by
any method or incorporated into another program without the express written
consent of XiTang and WuQi. This Information or any portion thereof remains
the property of XiTang and WuQi. The Information contained herein is believed
to be accurate and XiTang and WuQi assumes no responsibility or liability for
its use in any way and conveys no license or title under any patent or copyright
and makes no representation or warranty that this Information is free from patent
or copyright infringement.
"""
import os
import cv2
import json
import numpy as np
try:
  from PIL import Image as pil_image
except ImportError:
  pil_image = None

if pil_image is not None:
  _PIL_INTERPOLATION_METHODS = {
      'nearest': pil_image.NEAREST,
      'bilinear': pil_image.BILINEAR,
      'bicubic': pil_image.BICUBIC,
  }

def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):
  """Loads an image into PIL format.
  Arguments:
      path: Path to image file
      grayscale: Boolean, whether to load the image as grayscale.
      target_size: Either `None` (default to original size)
          or tuple of ints `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported. By default, "nearest" is used.
  Returns:
      A PIL Image instance.
  Raises:
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """


  img = cv2.imread(path,-1)

  if (np.max(img) - np.min(img) == 0):
    img[:,:] = 0
  elif 'light' in path or 'sznormal' in path or 'GMdatas' in path:
    #适用于物奇模组采集的数据
    #if np.min(img) <= 405:
    #  img[img <= 405] = 405
    mmin = np.min(img)
    mmax = np.max(img)
    img = (img - mmin) / (mmax - mmin)
  else:
    #适用于平板采集的数据
    img = img * 2
    img[img == 0] = 405
    mmin = np.min(img)
    mmax = np.max(img)
    img = (img - mmin) / (mmax - mmin)

  if len(img.shape) == 3:
    img = img[:, :, 0]


  return img

def to_one_hot(y, C):
    batch_size = len(y)
    return np.eye(C)[y.reshape(-1)].reshape(batch_size, -1, 1, 1)

def pad(train_data, size=(32, 32)):
    n, c, h, w = train_data.shape
    paded_data = np.zeros((n, c, size[0], size[1]))
    off_h = (size[0]-h)//2
    off_w = (size[1]-w)//2
    paded_data[:, :, off_h:size[0]-off_h, off_w:size[1]-off_w] = train_data
    return paded_data

class Data_Generator():
  """
  Arguments:
      n: Integer, total number of samples in the dataset to loop over.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seeding for data shuffling.
  """

  def __init__(self,
               data_dir,
               batch_size,
               shuffle,
               seed,
               n,
               filenames_to_class='filenames_to_class.json',
               target_size=(112,96),
               interpolation='bilinear',
               grayscale = False,
               one_hot = True,
               num_class = 1000):

    self.data_dir = data_dir
    with open(os.path.join(data_dir, filenames_to_class), 'r') as f:
      self.filenames_to_class = json.load(f)
    self.idx_to_fname = {}
    for idx, key in enumerate(self.filenames_to_class.keys()):
        self.idx_to_fname[idx] = key
    self.n = n
    self.batch_size = batch_size
    self.seed = seed
    self.shuffle = shuffle
    self.batch_index = 0
    self.total_batches_seen = 0
    self.index_array = None
    self.index_generator = self._flow_index()
    self.target_size = target_size
    self.interpolation = interpolation
    self.grayscale = grayscale
    if grayscale:
        self.channel = 1
    else:
        self.channel = 3
    self.one_hot = one_hot
    self.num_class = num_class
  def _set_index_array(self):
    self.index_array = np.arange(self.n)
    if self.shuffle:
      self.index_array = np.random.permutation(self.n)

  def __getitem__(self, idx):
    if idx >= len(self):
      raise ValueError('Asked to retrieve element {idx}, '
                       'but the Sequence '
                       'has length {length}'.format(idx=idx, length=len(self)))
    if self.seed is not None:
      np.random.seed(self.seed + self.total_batches_seen)
    self.total_batches_seen += 1
    if self.index_array is None:
      self._set_index_array()
    index_array = self.index_array[self.batch_size * idx:self.batch_size * (
        idx + 1)]
    return self._get_batches_of_transformed_samples(index_array)

  def __len__(self):
    return (self.n + self.batch_size - 1) // self.batch_size  # round up

  def on_epoch_end(self):
    self._set_index_array()

  def reset(self):
    self.batch_index = 0

  def _flow_index(self):
    # Ensure self.batch_index is 0.
    self.reset()
    while 1:
      if self.seed is not None:
        np.random.seed(self.seed + self.total_batches_seen)
      if self.batch_index == 0:
        self._set_index_array()

      current_index = (self.batch_index * self.batch_size) % self.n
      if self.n > current_index + self.batch_size:
        self.batch_index += 1
      else:
        self.batch_index = 0
      self.total_batches_seen += 1
      yield self.index_array[current_index:current_index + self.batch_size]

  def __iter__(self):
    return self

  def _get_batches_of_samples(self, index_array):
    batch_x = np.zeros(
        tuple([len(index_array)] + [self.channel] + list(self.target_size)), dtype=np.float32)
    batch_y = np.zeros((len(index_array),), dtype=np.int)

    batch_z = []

    for i, j in enumerate(index_array):
      #if "txt" in os.path.join(self.data_dir, self.idx_to_fname[j]):
      #  continue
      img = load_img(os.path.join(self.data_dir, self.idx_to_fname[j]),
                   target_size=self.target_size, interpolation=self.interpolation, grayscale=self.grayscale)

      #print(img, "shape", img.shape)
      img = np.array(img)
      if not self.grayscale:
          img = np.transpose(img, axes=(2,0,1))
      x = img
      batch_x[i] = x
      batch_z.append(self.idx_to_fname[j])
      #print(batch_x[i])
      if 'true' in self.idx_to_fname[j]:
        #print(self.idx_to_fname[j])

        batch_y[i] = 1
      else:
        batch_y[i] = 0
        #batch_y[i] = int(self.filenames_to_class[self.idx_to_fname[j]])

    #if self.one_hot:
    batch_y = to_one_hot(batch_y, self.num_class)
    batch_z = np.array(batch_z)
    return batch_x, batch_y, batch_z

  def next(self):
    """For python 2.x.
    Returns:
        The next batch.
    """
    index_array = next(self.index_generator)
    return self._get_batches_of_samples(index_array)

  def __next__(self):
    return self.next()

def data_generator_imagenet_partial(
               imagenet_dirname,
               batch_size,
               random_order,
               seed=100,
               n=1000,
               filenames_to_class='filenames_to_class.json',
               target_size=(112,96),
               interpolation='bilinear',
               grayscale = False,
               one_hot = True,
               num_class = 1000):

    dg = Data_Generator(data_dir=imagenet_dirname,batch_size=batch_size,shuffle=random_order,
                        seed=seed,n=n,filenames_to_class=filenames_to_class,
                        target_size=target_size,interpolation=interpolation,grayscale=grayscale,
                        one_hot=one_hot,num_class=num_class)

    while 1:
        batchx, batchy, batchz = next(dg)
        yield {'input':batchx,'output':batchy, 'filenames':batchz}


if __name__ == '__main__':
#    filenames, fclasses, subfilenames, subfclasses = get_filenames_to_class(r'J:\dataset\ILSVRC2012\train',
#                                                r'J:\01project\07ndk\ndk\examples\integer_encoding.json')
#    copy_to_dir(r'J:\01project\07ndk\ndk\examples\sub_imagenet_data',
#                subfilenames,
#                subfclasses)
    sg = data_generator_imagenet_partial(
                             imagenet_dirname=r'/home/kazyun/WQ5007/ndk_1.2.0/ndk/examples/liveness/test',
                             batch_size=24,
                             random_order=True)

    for i in range(1100):
        data_batch = next(sg)
        batchx = data_batch['input']
        batchy = data_batch['output']
        print(batchx.shape, batchy.shape)
