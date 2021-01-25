import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json 


# Custom data loader for Flickr8k dataset
class Loadf8k(data.Dataset):
  def __init__(self, root, vocab, split='train', transform=None):
    self.root = root
    self.vocab = vocab
    self.split = split
    self.transform = transform
    filename = root + '/flickr8k.json'
    self.dataset = json.load(open(filename, 'r'))['images']
    self.ids = []
    for i, d in enumerate(self.dataset):
      if d['split'] == split:
        self.ids += [(i, x) for x in range(len(d['sentences']))]

  def __getitem__(self, index):
    vocab = self.vocab
    root = self.root
    ann_id = self.ids[index]
    img_id = ann_id[0]
    caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
    path = 'images/' + self.dataset[img_id]['filename']

    image = Image.open(os.path.join(root, path)).convert('RGB')
    if self.transform is not None:
      image = self.transform(image)

    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    return image, target

  def __len__(self):
    return len(self.ids)


def collate_fn(data):
  # Sort a data list by caption length
  data.sort(key=lambda x: len(x[1]), reverse=True)
  images, captions = zip(*data)

  # Merge images (convert tuple of 3D tensor to 4D tensor)
  images = torch.stack(images, 0)

  # Merge captions (convert tuple of 1D tensor to 2D tensor)
  lengths = [len(cap) for cap in captions]
  targets = torch.zeros(len(captions), max(lengths)).long()
  for i, cap in enumerate(captions):
    end = lengths[i]
    targets[i, :end] = cap[:end]

  return images, targets, lengths


class Vocabulary(object):
  def __init__(self):
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 0
    
  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __call__(self, word):
    if word not in self.word2idx:
      return self.word2idx['<unk>']
    return self.word2idx[word]

  def __len__(self):
    return len(self.word2idx)
