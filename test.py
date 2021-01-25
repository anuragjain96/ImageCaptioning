import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from model import *
from utils import *
from PIL import Image


def loadImg(filename, transform):
  image = Image.open(filename).convert('RGB')
  image = image.resize([224, 224])
  image = transform(image).unsqueeze(0)
  print (image.shape)
  return image


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.255])])


# image for testing
#filename = 'data/f8k/images/3279025792_23bfd21bcc.jpg' 
filename = 'data/f8k/images/3393035454_2d2370ffd4.jpg'

vocab_file = 'data/f8k/f8k_vocab.pkl'
vocab = pickle.load(open(vocab_file, 'rb'))

encoder = Encoder(256).cuda()
decoder = Decoder(256, 512, len(vocab)).cuda()

# move models to val mode
encoder.eval()
decoder.eval()

encoder.load_state_dict(torch.load('encoder.ckpt'))
decoder.load_state_dict(torch.load('decoder.ckpt'))

img = loadImg(filename, transform)
img = img.cuda()
imgFeat = encoder(img)

captionIds = decoder.genCaption(imgFeat)
# move it to cpu
captionIds = captionIds.cpu().numpy()

caption = []

# mapping the idx to the actual words
for idx in captionIds:
  word = vocab.idx2word[idx]
  caption.append(word)
  if word == '<end>':
    break

print ('Image: ', filename)
sentence = ' '.join(caption)
print ('Generated Caption: ', sentence)

image = Image.open(filename)
plt.imshow(np.asarray(image))
