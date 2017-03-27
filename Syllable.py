from SoundObject import SoundObject
import matplotlib.pyplot as plt

class Syllable(SoundObject):

    def __init__(self, start, end, sound_file, species = 'Unassigned', label = 'Unassigned'):
        self.species = str(species)
        self.label = str(label)
        super(Syllable, self).__init__(start, end, sound_file)

    def __str__(self):
        out = super(Syllable, self).__str__() + '''
        Species: {0}
        Label: {1}
        '''.format(self.species, self.label)
        return(out)

    def plot_spectrogram(self, show):
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
