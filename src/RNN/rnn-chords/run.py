import os
import sys
import glob
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import normalise_song, generate_note_sequences, define_rnn_model, train_rnn_model, predict_next_note, create_midi_chords

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_utils import normalise_song_pretty, visualise_song, plot_distributions

def rnn_chord_train(number_of_epoch=100, train_model=True):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    midi_files = glob.glob("../../../data/classical-piano/*.mid")
    num_files = 10 

    all_notes = []
    for file in midi_files[:num_files]:
        normalised_notes = normalise_song(file)
        all_notes.extend(normalised_notes)
    
    vocabulary_size = len(set(all_notes))
    network_in, network_out = generate_note_sequences(notes=all_notes, sequence_length=100, vocabulary_size=vocabulary_size)
    model = define_rnn_model(network_in, vocabulary_size)

    if train_model:
        history = train_rnn_model(model, network_in, network_out, number_of_epoch=number_of_epoch)
        plt.plot(history.epoch, history.history['loss'], label='total loss')
        plt.show()
    else:
        try:
            model.load_weights(f'./weights_chords.hdf5')
        except:
            sys.exit("Weight File Not Found")


    pitchnames = sorted(set(item for item in all_notes))
    prediction_output = predict_next_note(model, all_notes, pitchnames, vocabulary_size)
    create_midi_chords(prediction_output)
    
    output_normalised_midi = normalise_song_pretty("./output.mid")
    visualise_song(output_normalised_midi)
    plot_distributions(output_normalised_midi)

if __name__ == "__main__":
    rnn_chord_train(number_of_epoch=200, train_model=False)
