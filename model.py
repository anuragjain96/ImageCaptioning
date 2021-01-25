import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# Using ResNet-50 model as the encoder module to encode the features
# of the input image
class Encoder(nn.Module):
  def __init__(self, embed_size):
    super(Encoder, self).__init__()
    resnet = models.resnet50(pretrained=True)
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.linear = nn.Linear(resnet.fc.in_features, embed_size)
    self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

  def forward(self, images):
    features = self.resnet(images)
    features = features.reshape(features.size(0), -1)
    features = self.bn(self.linear(features))
    return features



# using LSTM to generate the captions from the given 
# image feature representation
class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_caption_len=50):
    super(Decoder, self).__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.max_caption_length = max_caption_len

  def forward(self, features, captions, lengths):
    embeddings = self.embed(captions)
    embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    pack = pack_padded_sequence(embeddings, lengths, batch_first=True)

    hiddens, _ = self.lstm(pack)
    outputs = self.linear(hiddens[0])
    return outputs

  def genCaption(self, features, states=None):
    sampled_ids = torch.tensor([], dtype=torch.long).cuda()
    inputs = features.unsqueeze(1)

    for i in range(self.max_caption_length):
      hiddens, states = self.lstm(inputs, states)
      outputs = self.linear(hiddens.squeeze(1))
      _, predicted = outputs.max(1)
      sampled_ids = torch.cat([sampled_ids, predicted], dim=0)
      inputs = self.embed(predicted)
      inputs = inputs.unsqueeze(1)

    return sampled_ids
