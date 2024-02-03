import mido
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mido import MidiFile, MidiTrack, Message, MetaMessage
from mido.midifiles.meta import KeySignatureError
	
def ticks_to_seconds(ticks, tempo, ticks_per_beat):
	"""
	Converts MIDI ticks to seconds based on the specified tempo and ticks per beat.

	Parameters:
	- ticks (int): The number of ticks to be converted into seconds.
	- tempo (int): The tempo in microseconds per beat. 
	- ticks_per_beat (int): The number of ticks per beat, as defined in the MIDI file.

	Returns:
	- float: The duration in seconds corresponding to the given number of ticks.
	"""
	return ticks * tempo / (ticks_per_beat * 1000000)   

def seconds_to_ticks(seconds, tempo, ticks_per_beat):
	"""
	Converts seconds to MIDI ticks based on the specified tempo and ticks per beat.

	Parameters:
	- seconds (int): The number of seconds to be converted into ticks.
	- tempo (int): The tempo in microseconds per beat. 
	- ticks_per_beat (int): The number of ticks per beat, as defined in the MIDI file.

	Returns:
	- float: The number of ticks corresponding to the given duration in seconds.
	"""
	return int((seconds * ticks_per_beat * 1000000) / tempo)   

def normalise_song(midi_file):
	"""
	Normalizes MIDI file data into a structured numpy array format.

	This function processes a MIDI file object, extracting note events, including their start times,
	and durations, adjusting for tempo changes within the file. It normalizes the timing information
	from MIDI ticks to seconds, providing a consistent time format across different MIDI files
	irrespective of their original timing settings.

	Parameters:
	- midi_file (MidiFile): The MIDI file object to be processed. This object must have attributes
	such as 'tracks' and 'ticks_per_beat', and support iteration over MIDI messages within tracks.

	Returns:
	- numpy.ndarray: A numpy array where each row represents a note event. Columns include the note's
	start time in seconds, the MIDI note number (indicating the pitch), and the note's duration in
	seconds.
	"""
	mid = midi_file
	notes = []
	tempo = 500000  # Default MIDI tempo (500,000 microseconds per beat)
	ticks_per_beat = mid.ticks_per_beat

	for track in mid.tracks:
		current_time = 0  # Current time in ticks
		for msg in track:
			current_time += msg.time
			if msg.type == 'set_tempo':
				tempo = msg.tempo
			elif msg.type == 'note_on':
				start_time = ticks_to_seconds(current_time, tempo, ticks_per_beat)
				notes.append((msg.note, start_time, 0))  # Append note with start time
			elif msg.type == 'note_off':
				end_time = ticks_to_seconds(current_time, tempo, ticks_per_beat)
				for note in notes:
					if note[0] == msg.note and note[2] == 0:  # Find the note to update its duration
						notes.remove(note)
						duration = end_time - note[1]
						notes.append((msg.note, note[1], duration))
						break

	# Convert notes to a numpy array for plotting
	notes_array = np.array([(note[1], note[0], note[2]) for note in notes])
	return notes_array

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







### WIP (To be completed when model is coded and generates array)
	
# def get_number_of_tracks(midi_file):
# 	try:
# 		return int(len(midi_file.tracks))
# 	except KeySignatureError as e:
# 		return None, e
	
# def array_to_midi(notes_array, filename='output.mid', tempo=500000, ticks_per_beat=100, track_number=2):
# 	"""
# 	Converts an array of note events back into a MIDI file.

# 	Parameters:
# 	- notes_array: Array of note events, where each event is a tuple (start_time, pitch, duration).
# 	- filename: Name of the output MIDI file.
# 	- tempo: The tempo in microseconds per beat.
# 	- ticks_per_beat: Resolution of the MIDI file, in ticks per beat.
# 	"""
# 	mid = MidiFile(ticks_per_beat=ticks_per_beat)
# 	track_info = MidiTrack()
# 	mid.tracks.append(track_info)
# 	track_info.append(MetaMessage('set_tempo', tempo=tempo))

# 	if track_number == 3:
# 		track_right = MidiTrack()
# 		track_left = MidiTrack()   
# 		mid.tracks.append(track_right)
# 		mid.tracks.append(track_left)  
# 		track_right.append(MetaMessage('track_name', name='Right Hand'))
# 		track_right.append(MetaMessage('channel_prefix', channel=0, time=0))
# 		track_left.append(MetaMessage('track_name', name='Left Hand'))
# 		track_left.append(MetaMessage('channel_prefix', channel=1, time=0))

# 		cumulative_time = 0
# 		last_start_time = 0
# 		last_index = 0
# 		for start_time, pitch, duration in notes_array:
# 			duration_tick = seconds_to_ticks(duration, tempo, ticks_per_beat)
# 			note_on_time = 0 if cumulative_time == 0 else 0
# 			track_right.append(Message('note_on', channel=0, note=int(pitch), velocity=100, time=note_on_time))
# 			track_right.append(Message('note_off', channel=0, note=int(pitch), velocity=0, time=duration_tick))
# 			if last_start_time <= start_time:
# 				last_start_time = start_time
# 				last_index += 1
# 				print(start_time, last_start_time)
# 			elif last_start_time > start_time:
# 				track_right = track_right[:-2]
# 				break
# 			cumulative_time += duration_tick
# 		for _, pitch, duration in notes_array[last_index:]:
# 			duration_tick = seconds_to_ticks(duration, tempo, ticks_per_beat)
# 			note_on_time = 0 if cumulative_time == 0 else 0
# 			track_left.append(Message('note_on', channel=1, note=int(pitch), velocity=100, time=note_on_time))
# 			track_left.append(Message('note_off', channel=1, note=int(pitch), velocity=0, time=duration_tick))
# 			cumulative_time += duration_tick
		
# 	mid.save(filename)

# if track_number == 3:
#   track_right = MidiTrack()
#   track_left = MidiTrack()   
#   mid.tracks.append(track_right)
#   mid.tracks.append(track_left)  
#   track_right.append(MetaMessage('track_name', name='Right Hand'))
#   track_right.append(MetaMessage('channel_prefix', channel=0, time=0))
#   track_left.append(MetaMessage('track_name', name='Left Hand'))
#   track_left.append(MetaMessage('channel_prefix', channel=1, time=0))
# elif track_number == 2:
#     track_main = MidiTrack()
#     mid.tracks.append(track_main)
#     track_main.append(MetaMessage('track_name', name='Main Track'))
#     track_main.append(MetaMessage('channel_prefix', channel=0, time=0))

# cumulative_time = 0
# track = None
# for note in notes_array:
#     _, pitch, duration = note
#     duration_tick = seconds_to_ticks(duration, tempo, ticks_per_beat)
#     if track_number == 3: track = track_right if pitch >= center_split else track_left
#     elif track_number == 2: track = track_main
#     elif track is None: continue

#     note_on_time = 0 if cumulative_time == 0 else 0
#     # Add the note_on event
#     if track == track_right:
#       track_right.append(Message('note_on', channel=0, note=int(pitch), velocity=64, time=note_on_time))
#       track_right.append(Message('note_off', channel=0, note=int(pitch), velocity=0, time=duration_tick))
#     elif track == track_left:
#       track_left.append(Message('note_on', channel=1, note=int(pitch), velocity=64, time=note_on_time))
#       track_left.append(Message('note_off', channel=1, note=int(pitch), velocity=0, time=duration_tick))
#     elif track == track_main:
#       track_main.append(Message('note_on', channel=0, note=int(pitch), velocity=64, time=note_on_time))
#       track_main.append(Message('note_off', channel=0, note=int(pitch), velocity=0, time=duration_tick))
	
#     cumulative_time += duration_tick

# # Save the MIDI file
# mid.save(filename)

