# ImageCaptioning
Image captioning using simple encoder decoder modules. <br>
For Encoder a simple CNN module has been used: pre-trained ResNet-50. <br>
For Decoder a simple LSTM nework has been used.

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 3
* PyTorch 1.4.0
* TensorBoard
* torchvision
* tqdm
* matplotlib

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```
* By default the code runs on GPU.

## Dataset
Flickr8k dataset has been used. <br>
The vocabulary and the captions file has been added to the data/f8k directory. <br>
You need to save the flickr8k images in data/f8k/images/ directory.

## Training
```python
python train.py
```

You can change the no of epochs, learning rate and batch size inside the train.py file. <br>
Trained encoder and decoder models will be saved.

## Testing
```python
python test.py
```

For testing a sigle image has been taken and we print the generated caption for the image. <br>
You can change the test image name inside test.py file.
