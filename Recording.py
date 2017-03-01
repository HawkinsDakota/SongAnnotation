import re
from Syllable import Syllable
from progressbar import ProgressBar, UnknownLength


class Recording(object):

    def __init__(self, sound_file, grid_file, species = 'Unassigned'):
        self.__check_inputs(sound_file, grid_file, species)
        self.sound_file = sound_file
        self.grid_file = grid_file
        self.species = species
        self.syllables = []
        self.num_syllables = 0
        self.background = []
        self.num_background = 0

    def __str__(self):
        out = '''
              Species: %s
              Audio file: %s
              TextGrid file: %s
              Number of syllables: %i
              Total number of annotations: %i
              ''' % (self.species, self.sound_file,
                     self.grid_file, self.num_syllables,
                     self.num_syllables + self.num_background)
        return(out)

    def __check_inputs(self, sound_file, grid_file, species):
        if type(sound_file) != str:
            raise IOError('Expected file path for <sound_file>.')

        if type(grid_file) != str:
            raise IOError('Expected file path for <grid_file>.')

        if not grid_file.endswith('.TextGrid'):
            raise IOError('''File does not end with .TextGrid.
            File must be in .TextGrid format.''')

    def new_syllable(self, start, end, label):
        new_syllable = Syllable(start, end, self.sound_file, self.species, label)
        self.syllables.append(new_syllable)
        self.num_syllables += 1

    def new_background(self, start, end):
        new_background = Syllable(start, end, self.sound_file,
          species = 'None', label = 'background')
        self.background.append(new_background)
        self.num_background += 1

    def get_annotations(self, keep_background = True):
        '''
        Parses a .TextGrid file for syllable annotations
        '''
        with open(self.grid_file) as read_file:
            print('Parsing .TextGrid and Audio files.')
            bar = ProgressBar(max_value = UnknownLength)
            for i, line in enumerate(read_file):
                bar.update(i)
                if re.search('intervals \[[0-9]*\]', line) != None:
                    start_line = read_file.readline().strip()
                    end_line = read_file.readline().strip()
                    label_line = read_file.readline().strip()
                    start = float(start_line.split('=')[1])
                    end = float(end_line.split('=')[1])
                    label = re.sub(r'^"|"$', '', label_line.split('=')[1].strip())
                    if label != '':
                        label = self.species + '_' + label
                        self.new_syllable(start, end, label)
                    elif keep_background:
                        self.new_background(start, end)

if __name__ == "__main__":
    recording_test = Recording('Downloads/CATH1.WAV', 'Downloads/CATH1.TextGrid', 'CATH1')
    print(recording_test)
    recording_test.get_annotations(False)
    print(recording_test)
    for i in range(5):
        recording_test.syllables[i].plot_spectrogram()
