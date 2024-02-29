import pretty_midi
import numpy as np
import collections
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import tensorflow as tf

##################################
##### CHORD HELPER FUNCTIONS #####
##################################
def notes_to_chord_name(notes):
    sorted_notes = sorted(notes, key=lambda note: note.pitch)
    # Convert note numbers to note names and join with '-'
    return '-'.join([pretty_midi.note_number_to_name(note.pitch) for note in sorted_notes])

def find_chords_and_notes(instrument):
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
    notes = []
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Process each instrument to find notes and chords
    for instrument in midi_data.instruments:
        notes.extend(find_chords_and_notes(instrument))
    
    return notes

def generate_note_sequences(notes, sequence_length, vocabulary_size):
    assign_note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
    assign_int_to_note = {number: note for note, number in assign_note_to_int.items()}

    network_input = [[assign_note_to_int[notes[i+j]] for j in range(sequence_length)] for i in range(len(notes) - sequence_length)]
    network_output = [assign_note_to_int[notes[i+sequence_length]] for i in range(len(notes) - sequence_length)]
    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(vocabulary_size)
    network_output = utils.to_categorical(network_output)
    
    return network_input, network_output, assign_int_to_note


def create_midi_chords(prediction_output, file_path="ouput.mid"):
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
    pm.write(file_path)

    return pm


##############################
### PITCH HELPER FUNCTIONS ###
##############################
def normalise_song_pitch(midi_file):
	"""
	Normalizes MIDI file data into a structured DataFrame format using the pretty_midi library.

	This function processes a MIDI file, extracting note events It normalizes the timing information
	into seconds, providing a consistent time format across different MIDI files.

	Parameters:
	- midi_file_path (str): The path to the MIDI file to be processed.

	Returns:
	- pd.DataFrame: A pandas DataFrame array where each row represents a note event.
	"""
	pm = pretty_midi.PrettyMIDI(midi_file)
	notes = collections.defaultdict(list)

	# Sort the notes by start time
	for instrument in pm.instruments:
		sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
		prev_start = sorted_notes[0].start

		for note in sorted_notes:
			start = note.start
			end = note.end
			notes['pitch'].append(note.pitch)
			notes['start'].append(start)
			notes['end'].append(end)
			notes['step'].append(start - prev_start)
			notes['duration'].append(end - start)
			prev_start = start

	return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def normalise_pitch(notes, vocab_size):
    """
    Normalizes the pitch of the notes by the vocabulary size.

    Parameters:
        notes (tf.SymbolicTensor): An array of notes, where each note is represented as a vector of [pitch, step, duration].
        vocab_size (int): The size of the pitch vocabulary.

    Returns:
        np.ndarray: The normalized notes array.
    """
    return notes / [vocab_size, 1.0, 1.0]

def split_input_label(sequence, vocabulary_size):
    """
    Splits a sequence of notes into input features and labels for model training.

    Parameters:
        sequence (np.ndarray): A sequence of notes to be split.
        vocabulary_size (int): The size of the pitch vocabulary for normalization.

    Returns:
        tuple: A tuple containing the input sequence (with normalized pitch) and a dictionary of labels
               for 'pitch', 'step', and 'duration'.
    """
    input_sequence = sequence[:-1]
    label_sequence = sequence[-1]
    input_sequence = normalise_pitch(input_sequence, vocabulary_size)
    labels = {'pitch': label_sequence[0], 'step': label_sequence[1], 'duration': label_sequence[2]}
    return input_sequence, labels

def generate_note_sequences_pitch(all_normalized_notes, sequence_length, vocabulary_size=128):
    """
    Generates training sequences from normalized notes data.

    Parameters:
        all_normalized_notes (dict): A dictionary with keys 'pitch', 'step', 'duration', each mapping to a list of normalized note values.
        sequence_length (int): The length of the sequences to generate.
        vocabulary_size (int, optional): The size of the pitch vocabulary for normalization. Defaults to 128.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing tuples of input sequences and labels.
    """

    selected_attributes = ['pitch', 'step', 'duration']
    training_data = np.stack([all_normalized_notes[attribute] for attribute in selected_attributes], axis=1)
    dataset = tf.data.Dataset.from_tensor_slices(training_data)
    sequence_length += 1  # Adjust for target sequence length
    sequences = dataset.window(size=sequence_length, shift=1, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(sequence_length, drop_remainder=True))
    return sequences.map(lambda sequence: split_input_label(sequence, vocabulary_size), num_parallel_calls=tf.data.AUTOTUNE)

def array_to_midi_pretty(notes, output_midi_file_path="output.mid", instrument_name="Acoustic Grand Piano", velocity=100):
	"""
	Converts an array of note information back into a MIDI file using pretty_midi.

	Parameters:
	- notes (pd.DataFrame): An array where each row represents a note events.
	- output_midi_file_path (str): The path where the generated MIDI file will be saved.
	"""
	pm = pretty_midi.PrettyMIDI()
	instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

	prev_start = 0
	for i, note in notes.iterrows():
		start = float(prev_start + note['step'])
		end = float(start + note['duration'])
		note = pretty_midi.Note(
			velocity=velocity,
			pitch=int(note['pitch']),
			start=start,
			end=end,
		)
		instrument.notes.append(note)
		prev_start = start

	pm.instruments.append(instrument)
	pm.write(output_midi_file_path)


###############################
### VISUALIZATION FUNCTIONS ###
###############################
def visualise_song(notes):
	"""
	Visualizes a song's notes as a piano roll using an array of note events.

	Parameters:
	- notes (pandas.DataFrame): A DataFrame where each row represents a note event with five
	elements: the MIDI note number, the start time in seconds, the notes end time, the notes time between 
    the last note and the current note, and the note's duration in seconds.
	"""
	fig, ax = plt.subplots(figsize=(14, 8))
	for _, row in notes.iterrows():
		rect = plt.Rectangle((row['start'], row['pitch']), row['duration'], 1, color="blue", alpha=0.5)
		ax.add_patch(rect)

	# Adjusting limits and labels
	ax.set_xlim(0, np.max(notes['start'] + notes['duration']))
	ax.set_ylim(0, 128) # MIDI note numbers and frequencies
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('MIDI Note Number')
	ax.set_title('Piano Roll Visualization')
	plt.show()

def plot_distributions(notes, drop_percentile=2.5):
	"""
	Plots distributions of pitch, step, and duration from a notes array.

	Parameters:
	- notes (pd.DataFrame): A pandas Dataframe where each row represents a note event,
	with columns for start time, pitch, end time, step, and duration.
	- drop_percentile (float, optional): The percentile of step and duration values to exclude
	from the high end when plotting. Helps in focusing on the core distribution by removing
	outliers. Defaults to 2.5.
	"""
	plt.figure(figsize=[15, 5])
	# Plotting pitch distribution
	plt.subplot(1, 3, 1)
	sns.histplot(notes, x="pitch", bins=20, kde=False)
	# Plotting step distribution
	plt.subplot(1, 3, 2)
	max_step = np.percentile(notes['step'], 100 - drop_percentile)
	sns.histplot(notes[notes['step'] <= max_step], x="step", bins=np.linspace(0, max_step, 21), kde=False)
	# Plotting duration distribution
	plt.subplot(1, 3, 3)
	max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
	sns.histplot(notes[notes['duration'] <= max_duration], x="duration", bins=np.linspace(0, max_duration, 21), kde=False)
	plt.tight_layout()
	plt.show()