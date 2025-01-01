# AI-Powered Music Generation

This project demonstrates an AI-powered music generation system using Recurrent Neural Networks (RNNs) with LSTM layers. The system is capable of composing original music based on the patterns it learns from the provided dataset of audio files.

## Installation

First, install the necessary packages:

```bash
pip install numpy pandas tensorflow music21 pydub librosa
```
## Usage 
### Step 1: Import Libraries

The first step is to import the necessary libraries for this project:
```python
import numpy as np
import pandas as pd
import os
from music21 import converter, instrument, note, chord, stream
from pydub import AudioSegment
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
```
### Step 2: Convert MP3 to WAV
Convert MP3 files to WAV format for easier processing:
```python
# Example code to convert MP3 to WAV
from pydub import AudioSegment

sound = AudioSegment.from_mp3("input.mp3")
sound.export("output.wav", format="wav")
```
### Additional Steps

The notebook continues with further steps to preprocess the audio data, train the model, and generate music. These steps include:

- **Preprocessing**: Extracting musical notes and chords from the audio files.
- **Model Training**: Setting up and training the LSTM model.
- **Music Generation**: Using the trained model to generate new music sequences.
For detailed code and explanations, please refer to the notebook.

## Project Structure
- `CodeAlpha_Music_Generation_with_AI.ipynb`:  The main notebook containing the full project
- `data/`: Directory to store input audio files (MP3 format).
- `output`/: Directory to save converted WAV files and generated music.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License 
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
This project utilizes the following libraries:

- numpy
- pandas
- tensorflow
- music21
- pydub
- librosa

Special thanks to the developers of these libraries for their contributions to the open-source community.

## Contact
If you have any questions or feedback, feel free to reach out to me at [souhail.mah@gmail.com].

