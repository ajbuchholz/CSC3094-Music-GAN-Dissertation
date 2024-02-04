import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def normalise_song(midi_file):
    """
    Normalizes MIDI file data into a structured numpy array format using the pretty_midi library.

    This function processes a MIDI file, extracting note events, including their start times,
    and durations, adjusting for tempo changes within the file. It normalizes the timing information
    into seconds, providing a consistent time format across different MIDI files.

    Parameters:
    - midi_file_path (str): The path to the MIDI file to be processed.

    Returns:
    - numpy.ndarray: A numpy array where each row represents a note event. Columns include the note's
    start time in seconds, the MIDI note number (indicating the pitch), and the note's duration in
    seconds.
    """
    notes = []
    for instrument in midi_file.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            duration = end_time - start_time
            notes.append((start_time, note.pitch, duration))
    notes_array = np.array(notes, dtype=float)
    return notes_array

def array_to_midi(notes_array, output_midi_file_path="output.mid"):
    """
    Converts an array of note information back into a MIDI file using pretty_midi.

    Parameters:
    - notes_array (numpy.ndarray): An array where each row represents a note event with
      columns for the note's start time in seconds, the MIDI note number (indicating the pitch),
      and the note's duration in seconds.
    - output_midi_file_path (str): The path where the generated MIDI file will be saved.
    """
    midi_data = pretty_midi.PrettyMIDI()
    # Create an Instrument instance | 0 refers to the program number for Acoustic Grand Piano.
    instrument = pretty_midi.Instrument(program=0)
    
    for note_info in notes_array:
        start_time, pitch, duration = note_info
        end_time = start_time + duration
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)
    midi_data.instruments.append(instrument)
    midi_data.write(output_midi_file_path)

def visualise_song(notes_array):
	"""
	Visualizes a song's notes as a piano roll using a numpy array of note events.

	Parameters:
	- notes_array (numpy.ndarray): A numpy array where each row represents a note event with three
	elements: the start time in seconds, the MIDI note number, and the note's duration in seconds.
	"""
	fig, ax = plt.subplots(figsize=(14, 8))
	for note in notes_array:
		rect = plt.Rectangle((note[0], note[1]), note[2], 1, color="blue", alpha=0.5)
		ax.add_patch(rect)

	ax.set_xlim(0, np.max(notes_array[:,0] + notes_array[:,2]))
	ax.set_ylim(0, 128) # https://inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('MIDI Note Number')
	ax.set_title('Piano Roll Visualization')
	plt.show()

def plot_distributions(notes_array, drop_percentile=2.5):
    """
    Plots distributions of pitch, step, and duration from a notes array.

    Parameters:
    - notes_array (np.ndarray): A numpy array where each row represents a note event,
      with columns for start time, pitch, and duration.
    - drop_percentile (float, optional): The percentile of step and duration values to exclude
      from the high end when plotting. Helps in focusing on the core distribution by removing
      outliers. Defaults to 2.5.
    """
    notes = pd.DataFrame(notes_array, columns=['start_time', 'pitch', 'duration'])
    notes['step'] = notes['start_time'].diff().fillna(0)
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