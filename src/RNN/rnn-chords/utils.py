import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras import utils


def notes_to_chord_name(notes):
    """
    Converts a list of pretty_midi Note objects to a chord name.

    Args:
        notes (list): A list of pretty_midi Note objects.

    Returns:
        str: The chord name formed by converting note numbers to note names and joining with '-'.
    """
    sorted_notes = sorted(notes, key=lambda note: note.pitch)
    # Convert note numbers to note names and join with '-'
    return '-'.join([pretty_midi.note_number_to_name(note.pitch) for note in sorted_notes])

def find_chords_and_notes(instrument):
    """Finds and groups notes and chords.

    Args:
        instrument (pretty_midi.Instrument): The instrument containing the notes.

    Returns:
        list: A list of chords and notes found in the instrument.
    """
    chords_and_notes = []
    sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    
    current_start_time = None
    current_group = []
    for note in sorted_notes:
        if note.start != current_start_time:
            if current_group:
                if len(current_group) == 1:
                    chords_and_notes.append(pretty_midi.note_number_to_name(current_group[0].pitch))
                else:
                    chords_and_notes.append(notes_to_chord_name(current_group))
                current_group = []
            current_start_time = note.start
        current_group.append(note)
    
    if current_group:
        if len(current_group) == 1:
            chords_and_notes.append(pretty_midi.note_number_to_name(current_group[0].pitch))
        else:
            chords_and_notes.append(notes_to_chord_name(current_group))
    
    return chords_and_notes

def normalise_song(midi_file_path):
    """
    Normalizes a song from a MIDI file into a list of notes and chords.
    
    Parameters:
        midi_file_path (str): The path to the MIDI file.
    
    Returns:
        list: A list of notes and chords in the song.
    """
    notes = []
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Process each instrument to find notes and chords
    for instrument in midi_data.instruments:
        notes.extend(find_chords_and_notes(instrument))
    
    return notes

def generate_note_sequences(notes, sequence_length, vocabulary_size):
    """
    Generate input and output sequences for a recurrent neural network (RNN) model.

    Args:
        notes (list): List of notes.
        sequence_length (int): Length of each input sequence.
        vocabulary_size (int): Size of the vocabulary (number of unique notes).

    Returns:
        tuple: A tuple containing the input sequences and the output sequences.
            - network_in (numpy.ndarray): Input sequences for the RNN model.
            - network_out (numpy.ndarray): Output sequences for the RNN model.
    """
    assign_note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}

    network_in = [[assign_note_to_int[notes[i+j]] for j in range(sequence_length)] for i in range(len(notes) - sequence_length)]
    network_out = [assign_note_to_int[notes[i+sequence_length]] for i in range(len(notes) - sequence_length)]
    network_in = np.reshape(network_in, (len(network_in), sequence_length, 1)) / float(vocabulary_size)
    network_out = utils.to_categorical(network_out)
    
    return network_in, network_out


def define_rnn_model(network_in, vocabulary_size):
    """
    Defines and compiles an RNN model for generating music chords.

    Args:
        network_in (numpy.ndarray): Input data for the RNN model.
        vocabulary_size (int): Size of the vocabulary (number of unique chords).

    Returns:
        tf.keras.models.Sequential: Compiled RNN model.

    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(network_in.shape[1], network_in.shape[2]), recurrent_dropout=0.25, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.25))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(vocabulary_size))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.summary()

    return model

def train_rnn_model(model, network_in, network_out, number_of_epoch):
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model_weights/checkpoint_{epoch}.hdf5',
        save_weights_only=True,
        monitor='loss', 
        mode='min',
        save_best_only=True
    )
    return model.fit(network_in, network_out, epochs=number_of_epoch, batch_size=128, callbacks=callback)
    

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

    num_predictions = 200
    for _ in range(num_predictions):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(vocabulary_size)
        prediction = model.predict(prediction_input, verbose=1)
        index = np.argmax(prediction)
        prediction_output.append(assign_int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def create_midi_chords(prediction_output):
    """
    Creates a MIDI file from a list of chord predictions.

    Args:
        prediction_output (list): A list of chord predictions.

    Returns:
        pretty_midi.PrettyMIDI: The PrettyMIDI object representing the created MIDI file.
    """
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    offset = 0 

    for pattern in prediction_output:
        # Split pattern into notes if it's a chord, otherwise treat as a single note
        elements = pattern.split('-') if '-' in pattern else [pattern]
        
        for el in elements:
            note_number = int(el) if el.isdigit() else pretty_midi.note_name_to_number(el)
            
            note = pretty_midi.Note(
                velocity=100, 
                pitch=note_number,
                start=offset,
                end=offset + 0.3  # Note duration
            )
            piano.notes.append(note)
        
        offset += 0.3

    pm.instruments.append(piano)
    pm.write('output.mid')

    return pm
