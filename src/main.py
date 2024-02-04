import os
import pretty_midi
from utils import visualise_song, normalise_song, array_to_midi, plot_distributions

def experiment_1():
    """
    Train RNN model to predict next note. Once model is trained provide it was some starting notes and let it complete the song
    """
    mid = pretty_midi.PrettyMIDI('../data/classical-piano/2inE.mid')
    normalized_song = normalise_song(mid)

experiment_1()