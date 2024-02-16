# CSC3094-Music-GAN-Dissertation

## Installation/Running
1. `python3 -m venv venv` | Create the Virtual Environment
2. `source venv/bin/activate` | Activate the Virtual Environment
3. `pip install -r requirements.txt` | Download the Required Dependencies
4. `python src/main.py`

## Goals
**Goal #1**: Create a Recurrent Neural Network (RNN) that can “accurately” predict the next note in a sequence. Once trained, the user will provide the model with some starting notes, and the model will select the best notes and timings to complete the song. This should ultimately generate music that has some form of harmonic resonance and/or sounds like it could be a “real” music piece. \
**Goal #2**: Create a GAN using a generator and discriminator that “compete” against one another to create realistic musical pieces. This also includes fine-tuning the hyperparameters and experimenting with different forms of RNNs such as Gated Recurrent Unit (GRU) or Long Short-Term Memory (LSTM), etc. \
**Goal #3**: Modify the GAN model in Goal #3 to work with multiple instruments and genres of music. Again, with the end goal being a “realistic” composition or musical piece. Sidenote: In goals #1 and #2, the only instrument being used is a piano, and the genre is classical. \
*Goal #4*: Experiment with the normalisation of music data into other methods other than the original array of [pitch, step, duration], etc. See if a different form of inputting the data into the model yields better results.

## Dataset Information
"The ADL Piano MIDI is a dataset of 11,086 piano pieces from different genres. This dataset is based on the Lakh MIDI dataset, which is a collection on 45,129 unique MIDI files that have been matched to entries in the Million Song Dataset."

`adl-piano-midi.zip` | ZIP file of all the MIDI files \
`adl-piano-midi/` | Directory containing the unzipped contents of the ZIP file \
`classical-piano/` | Directory containing all the MIDI classical piano files (training set)

### Dataset References
```
[1] L. N. Ferreira, L. H. S. Lelis, and J. Whitehead, “Computer-Generated Music for Tabletop Role-Playing Games,” arXiv.org, Aug. 16, 2020. https://arxiv.org/abs/2008.07009.
[2] Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.
```

## RNN Models
**Goals #1 and #4**
1. First experiment with creating an RNN model. The `rnn_pitch` directory utilises the original idea of normalising the data into an array of [pitch, step, duration]. Then train a model to predict the next note in a sequence of 25 notes. However, the downsides were that the model was unable to effectively play chords, so the song was, in essence, one note being pressed at a time.
2. The second experiment was to normalise the input data differently so the model could understand the difference between notes and chords. The `rnn_chords` directory uses a different format for normalising our data in the form of notes and chords. Similarly, the model is trained to predict the next note or chord in a sequence.

## GAN Model
**Goals #2, #3, and #4**