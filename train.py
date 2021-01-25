import torch
import numpy as np
import os
import pickle
from utils import *
from model import *
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.255])])


root = 'data/f8k'
vocab_file = 'data/f8k/f8k_vocab.pkl'
vocab = pickle.load(open(vocab_file, 'rb'))

dataset = Loadf8k(root, vocab, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64,
                                          shuffle=True, pin_memory=True,
                                          num_workers=16, collate_fn=collate_fn)

encoder = Encoder(256).cuda()
decoder = Decoder(256, 512, len(vocab)).cuda()

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

total_epochs = 10

# training the encoder and decoder module
for epoch in range(total_epochs):
  train_bar = tqdm(data_loader)
  totalLoss = 0.0
  totalNum = 0

  for images, captions, lengths in train_bar:

    images, captions = images.cuda(), captions.cuda()
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    imgFeat = encoder(images)
    output  = decoder(imgFeat, captions, lengths)

    loss = criterion(output, targets)
    totalLoss += loss
    totalNum += images.size(0)

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    optimizer.step()

    train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch+1, total_epochs, totalLoss / totalNum))

  # saving the models
  torch.save(encoder.state_dict(), 'encoder.ckpt')
  torch.save(decoder.state_dict(), 'decoder.ckpt')
