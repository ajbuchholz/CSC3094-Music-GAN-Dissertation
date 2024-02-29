# CSC3094-Music-GAN-Dissertation

## Installation/Running
1. `python3 -m venv venv` | Create the Virtual Environment
2. `source venv/bin/activate` | Activate the Virtual Environment
3. `pip install -r requirements.txt` | Download the Required Dependencies
4. `pip install keras --upgrade` | Make sure it's Keras 3.0+

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
1. First experiment with creating an RNN model. The `rnn_pitch_model` Python file utilises the original idea of normalising the data into an array of [pitch, step, duration]. Then train a model to predict the next note in a sequence of 100 notes. However, the downsides were that the model was unable to effectively play chords, so the song was, in essence, one note being pressed at a time.
2. The second experiment was to normalise the input data differently so the model could understand the difference between notes and chords. The `rnn_chords_model` Python file uses a different format for normalising our data in the form of notes and chords. Similarly, the model is trained to predict the next note or chord in a sequence.

## GAN Model
WIP

## Tranformer Model
WIP