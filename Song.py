from SyllableCollection import SyllableCollection
from SoundObject import SoundObject

class Song(SyllableCollection, SoundObject):

    def __init__(self, syllables, start, end, sound_file, preprocess = True,
    fmin = 0, fmax = None):
        SyllableCollection.__init__(self, syllables)
        SoundObject.__init__(self, start, end, sound_file, preprocess,
          fmin, fmax)

if __name__ == "__main__":
    from Recording import Recording
    recording_test = Recording('Downloads/CATH1.wav', 'Downloads/CATH1.TextGrid', 'CATH1')
    recording_test.get_annotations(False)
    song_test = Song(recording_test.syllables[0:26],
                     recording_test.syllables[0].start,
                     recording_test.syllables[26].end,
                     'Downloads/CATH1.wav',
                     True,
                     0,
                     None)
    #song_test.plot_spectrogram(True)
    print(song_test[0])
    print(song_test.syllables[1:2])
