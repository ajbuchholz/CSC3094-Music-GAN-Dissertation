# CSC3094-Music-GAN-Dissertation

## Installation/Running
1. `python3 -m venv venv` | Create the Virtual Environment
2. `source venv/bin/activate` | Activate the Virtual Environment
3. `pip install -r requirements.txt` | Download the Required Dependencies
4. `pip install keras --upgrade` | Make sure it's Keras 3.0+

## Dataset Information
"The ADL Piano MIDI is a dataset of 11,086 piano pieces from different genres. This dataset is based on the Lakh MIDI dataset, which is a collection on 45,129 unique MIDI files that have been matched to entries in the Million Song Dataset."

`classical-piano/` | Directory containing all the MIDI classical piano files. [1] [2]
`maestro/` | Directory containing the Maestro MIDI dataset. [3]
`mozart-multi/` | Directory containing multi-track MIDI dataset. 

### Dataset References
```
[1] L. N. Ferreira, L. H. S. Lelis, and J. Whitehead, “Computer-Generated Music for Tabletop Role-Playing Games,” arXiv.org, Aug. 16, 2020. https://arxiv.org/abs/2008.07009.
[2] Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.
[3] C. Hawthorne, et al. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." In International Conference on Learning Representations, 2019.
```

## RNN Models
1. `rnn_pitch_model` | [pitch, step, duration] | Classical Piano Dataset
2. `rnn_chords_model.py` | Note and Chords | Classical Piano Dataset

## GAN Models
1. `gan_model.py` | Notes and Chords, Frequency, and Note Events | Classical Piano Dataset 
2. `gan_model_multiple.py` | Notes and Chords for Multi-Track (multiple instruments) MIDI files | Mozart MIDI Files

## Tranformer Model
1. `transformer/transformer_model.py` | Notes and Chords (Seq2Seq) | Maestro & Classical Piano