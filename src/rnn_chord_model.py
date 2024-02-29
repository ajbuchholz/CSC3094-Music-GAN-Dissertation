import sys
import glob
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from utils import normalise_song, generate_note_sequences, create_midi_chords, normalise_song_pitch, visualise_song, plot_distributions

NUMBER_OF_EPOCHS = 200
SEQUENCE_LENGTH = 100
TRAIN_MODEL = False
NUMBER_FILES_TO_PARSE = 10
NUMBER_OF_PREDICTIONS = 100
BATCH_SIZE = 128

def define_rnn_model(network_in, vocabulary_size):
    """
    Defines and compiles an RNN model for generating music chords.

    Args:
        network_in (numpy.ndarray): Input data for the RNN model.
        vocabulary_size (int): Size of the vocabulary (number of unique chords).

    Returns:
        models.Sequential: Compiled RNN model.

    """
    model = models.Sequential()
    model.add(layers.LSTM(256, input_shape=(network_in.shape[1], network_in.shape[2]), recurrent_dropout=0.25, return_sequences=True))
    model.add(layers.LSTM(256, return_sequences=True, recurrent_dropout=0.25))
    model.add(layers.LSTM(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(vocabulary_size))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.summary()

    return model

def train_rnn_model(model, network_in, network_out, number_of_epoch):
    callback = callbacks.ModelCheckpoint(
        filepath='rnnchord_checkpoints/checkpoint_{epoch}.hdf5',
        save_weights_only=True,
        monitor='loss', 
        mode='min',
        save_best_only=True
    )
    return model.fit(network_in, network_out, epochs=number_of_epoch, batch_size=BATCH_SIZE, callbacks=callback)
    

def predict_next_note(model, notes, pitchnames, vocabulary_size):
    """
    Predicts the next note in a sequence using a trained model.

    Parameters:
    model (keras.Model): The trained model used for prediction.
    notes (list): The list of notes used as input for prediction.
    pitchnames (list): The list of pitch names used for mapping notes to integers.
    vocabulary_size (int): The size of the vocabulary (number of unique pitch names).

    Returns:
    list: The predicted sequence of notes.
    """
    assign_note_to_int = {note: num for num, note in enumerate(pitchnames)}
    network_in = []
    sequence_length = 100
    for i in range(len(notes) - sequence_length):
        sequence_in = [assign_note_to_int[note] for note in notes[i:i + sequence_length]]
        network_in.append(sequence_in)

    start = np.random.randint(0, len(network_in) - 1)
    assign_int_to_note = {number: note for number, note in enumerate(pitchnames)}
    pattern = list(network_in[start])
    prediction_output = []

    for _ in range(NUMBER_OF_PREDICTIONS):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(vocabulary_size)
        prediction = model.predict(prediction_input, verbose=1)
        index = np.argmax(prediction)
        prediction_output.append(assign_int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

if __name__ == '__main__':
    midi_files = glob.glob("../data/classical-piano/*.mid")

    all_notes = []
    for file in midi_files[:NUMBER_FILES_TO_PARSE]:
        normalised_notes = normalise_song(file)
        all_notes.extend(normalised_notes)
    
    vocabulary_size = len(set(all_notes))
    network_in, network_out, _ = generate_note_sequences(notes=all_notes, sequence_length=SEQUENCE_LENGTH, vocabulary_size=vocabulary_size)
    model = define_rnn_model(network_in, vocabulary_size)

    if TRAIN_MODEL:
        history = train_rnn_model(model, network_in, network_out, number_of_epoch=NUMBER_OF_EPOCHS)
        plt.plot(history.epoch, history.history['loss'], label='total loss')
        plt.show()
    else:
        try:
            model.load_weights('final_weights/weights_chord.hdf5')
        except:
            sys.exit("Weight File Not Found")

    pitchnames = sorted(set(item for item in all_notes))
    prediction_output = predict_next_note(model, all_notes, pitchnames, vocabulary_size)
    create_midi_chords(prediction_output, "output-rnn-chord.mid")
    
    output_normalised_midi = normalise_song_pitch("./output-rnn-chord.mid")
    visualise_song(output_normalised_midi)
    plot_distributions(output_normalised_midi)
