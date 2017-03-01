from SoundObject import SoundObject

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

if __name__ == '__main__':
    test = Syllable(start = 11.844455261385376,
                    end = 12.081455063757392,
                    sound_file = 'Downloads/CATH1.WAV',
                    species = 'CATH',
                    label = 'CATH_yea')

    print(test)
    test.plot_spectrogram()
