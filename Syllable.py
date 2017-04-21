from SoundObject import SoundObject
import matplotlib.pyplot as plt

class Syllable(SoundObject):
    """ Class to represent syllable instances in bird songs.

    Syllable class inherits from the more general SoundObject class.

    Args:
        species (str, optional): String denoting which species the syllable
            belongs to. If no argument is provided, defaults to 'Unassigned'
        label (str, optional): Label of syllable type. Set to 'None' if no
            label is provided.

    Attributes:
        species (str, optional): String denoting species the syllable belongs
            to.
        label (str, optional): Label of syllable type.
        start (float): Start time in seconds of syllable in audio file.
        end (float): End time in seconds of syllable in audio file.
        duration (float): Duration in seconds of syllable.
        sound_file (str): String of audio file syllable was read from. Should be
            .wav file.
        fmax (float, optional): Maximum frequency in Hz to keep. All frequencies
            above the threshold will be removed.
        fmin (float, optinal): Minimum frequency in Hz to keep. All frequenceis
            below the threshold will be removed.
        sound (:numpy.array:float): Numpy array containing sound data for the
            syllable.
        sample_rate (integer): Sample rate used to sample `sound` from
            `sound_file`.
        bin_size (integer): Number of sample bins required for 1 ms of time.
        spectrogram (:numpy.array:float): A 128 x (`duration`*`bin_size`) numpy
            array containing decibel values of the sound in frequency bins
            from `fmin` = `fmax` over the duration of the syllable.
    """

    def __init__(self, start, end, sound_file,
                 species='Unassigned', label='Unassigned'):
        self.species = str(species)
        self.label = str(label)
        super(Syllable, self).__init__(start, end, sound_file)

    def __str__(self):
        out = super(Syllable, self).__str__() + '''
        Species: {0}
        Label: {1}
        '''.format(self.species, self.label)
        return(out)

    def __eq__(self, other):
        if not isinstance(other, Syllable):
            raise TypeError("Can't compare type {0} with Syllable".format(type(other)))
        label_eq = self.label == other.label
        species_eq = self.species == other.species
        return(all([label_eq, species_eq, super(Syllable, self).__eq__(other)]))

    def plot_spectrogram(self, show=True):
        """ Plot the spectrogram of the Syllable.

        Plots the spectrogram associated with each Syllable instance. Inherrited
        from SoundObject class.

        Args:
            show (boolean, optional): value that determines whether to show
            generated plot. Default value is True.

        Returns:
            None.
        """
        super(Syllable, self).plot_spectrogram(False)
        plot_title = '%s from [%0.2f - %0.2f] in %s' % (self.label,
          self.start, self.end, self.species)
        plt.title(plot_title)
        if show:
            plt.show()

if __name__ == '__main__':
    test = Syllable(start = 11.844455261385376,
                    end = 12.081455063757392,
                    sound_file = 'Downloads/CATH1.WAV',
                    species = 'CATH',
                    label = 'CATH_yea')

    print(test)
    print(isinstance(test, Syllable))
    test.plot_spectrogram(True)
