import os
import mido
from utils import visualise_song, normalise_song


### Useful for Debugging
# def print_midi_messages(input_filename='output.mid', output_filename='midi_messages.txt'):
#     # Load the MIDI file
#     midi_file = mido.MidiFile(input_filename)

#     with open(output_filename, 'w') as file:
#         for i, track in enumerate(midi_file.tracks):
#             track_name = track.name if hasattr(track, 'name') else f'Track {i}'
#             file.write(f"Track {i}: {track_name}\n")
#             for msg in track:
#                 file.write(f"{msg}\n")

mid = mido.MidiFile('../data/classical-piano/2inE.mid')
normalized_song = normalise_song(mid)
visualise_song(normalized_song)