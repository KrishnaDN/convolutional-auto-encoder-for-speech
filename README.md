# convolutional-auto-encoder-for-speech
This model implements auto-encoder for speech data using deep convolution neural networks in Pytorch.
We use data from audioset project which downloads has audio data from youtube videos. We first download youtube videos and then we extract the audio files and finally we downsample them to 16KHz. The model takes in 2sec spectrogram (200x257) as input to a Encoder of CNN and then projects it to 256 dimensional vector. We then  recostruct the spectrogram using Decoder CNN

## Getting Started


### Prerequisites
First We need to install ffmpeg and youtube-dl
For Linux OS
```
sudo apt-get install ffmpeg
sudo apt-get install youtube-dl
```
For Mac OS
```
brew install ffmpeg
brew install youtube-dl
```

### Installing python packages
Install the required python packages as follows
```
pip install -r requirements
```
## Downloading the data
First we need to download the data from youtube and extract audio. To do this run the script as follows
```
python data_processing.py
```
This will download the videos and saves the audio files in "data_files" folder

### Split train and test
```
python create_train_test.py
```

### Training the model
To train the auto-encoder model use
```
python trainig.py
```
