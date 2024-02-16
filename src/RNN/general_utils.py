import collections
import pretty_midi
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalise_song_pretty(midi_file):
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