import os
import glob
import sys
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import generate_note_sequences, define_rnn_model, predict_next_note, train_rnn_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_utils import visualise_song, normalise_song_pretty, array_to_midi_pretty, plot_distributions

def rnn_pitch_train(number_of_epoch=100, train_model=True):
    """
    Train LSTM Model to Predict 26th note in a sequence. The model utilises pitch, step, and duration.
    """
    # Set seed
    if train_model:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Load our MIDI files from the directory
    midi_files = glob.glob("../../../data/classical-piano/*.mid")
    num_files = 10

    # Normalise our dataset
    normalised_notes = []
    for file in midi_files[:num_files]:
        notes = normalise_song_pretty(file)
        normalised_notes.append(notes)
    all_normalised_notes = pd.concat(normalised_notes)

    # Set the sequence length for each example
    key_order = ['pitch', 'step', 'duration']
    sequence_length = 25
    vocabulary_size = 128
    sequence_song_dataset = generate_note_sequences(all_normalised_notes, sequence_length, vocabulary_size)

    # Define RNN Model
    batch_size = 64
    buffer_size = len(all_normalised_notes) - sequence_length  # the number of items in the dataset
    training_song_dataset = sequence_song_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
    model = define_rnn_model(sequence_length=sequence_length)
    
    if train_model:
        # Train Model
        history = train_rnn_model(model, training_song_dataset, number_of_epoch=number_of_epoch)

        plt.plot(history.epoch, history.history['loss'], label='total loss')
        plt.show()
    else:
        # Load Model
        try:
            model.load_weights(f'./weights_pitch.hdf5')
        except:
            sys.exit("Weight File Not Found")

    # Generate Song
    num_predictions = 100
    input_notes = np.stack([all_normalised_notes[key] for key in key_order], axis=1)[:sequence_length] / np.array([vocabulary_size, 1, 1])
    generated_notes = []
    start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature=2.0, subtract_step=0.5, subtract_duration=0.0)
        start += step
        end = start + duration
        generated_note = (pitch, step, duration, start, end)
        generated_notes.append(generated_note)
        input_notes = np.vstack([input_notes[1:], np.array(generated_note[:3])])

    generated_notes = pd.DataFrame(generated_notes, columns=[*key_order, 'start', 'end'])

    # Save Generate Song
    visualise_song(generated_notes)
    plot_distributions(generated_notes)
    array_to_midi_pretty(generated_notes)

if __name__ == "__main__":
    rnn_pitch_train(number_of_epoch=500, train_model=False)
