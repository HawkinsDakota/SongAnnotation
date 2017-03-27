import re
from Syllable import Syllable
from SyllableCollection import SyllableCollection
from Song import Song
from progressbar import ProgressBar, UnknownLength

class Recording(object):

    def __init__(self, sound_file, grid_file, species = 'Unassigned',
      preprocess = True, fmin = 0, fmax = None):
        self.__check_inputs(sound_file, grid_file, species)
        self.preprocess = preprocess
        self.fmin = fmin
        self.fmax = fmax
        self.sound_file = sound_file
        self.grid_file = grid_file
        self.species = species
        self.syllables = SyllableCollection()
        self.num_syllables = 0
        self.songs = []
        self.num_songs = 0
        self.background = []
        self.num_background = 0

    def __str__(self):
        out = '''
              Species: %s
              Audio file: %s
              TextGrid file: %s
              Number of syllables: %i
              Number of songs: %i
              Total number of annotations: %i
              ''' % (self.species, self.sound_file,
                     self.grid_file, self.num_syllables, self.num_songs,
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
        self.syllables.add_syllable(new_syllable)
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
            song_start = 0
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
                        current_syllable = self.syllables[self.num_syllables - 1]
                        past_syllable = self.syllables[self.num_syllables - 2]
                        if (self.num_syllables > 1 and current_syllable.end - past_syllable.start > 1):
                            self.add_song(self.syllables[song_start:(self.num_syllables - 1)],
                                          self.syllables[song_start].start,
                                          current_syllable.end,
                                          self.sound_file,
                                          self.preprocess,
                                          self.fmin,
                                          self.fmax)
                            song_start = self.num_syllables
                    elif keep_background:
                        self.new_background(start, end)
            self.add_song(self.syllables[song_start:(self.num_syllables - 1)],
                          self.syllables[song_start].start,
                          self.syllables[self.num_syllables - 1].end,
                          self.sound_file,
                          self.preprocess,
                          self.fmin,
                          self.fmax)

    def add_song(self, song_syllables, song_start, song_end, sound_file,
      preprocess, fmin, fmax):
        self.songs.append(Song(
           song_syllables,
           song_start,
           song_end,
           self.sound_file,
           self.preprocess,
           self.fmin,
           self.fmax))
        self.num_songs += 1


    def list_syllables(self):
        label_list = ['']*self.num_syllables
        for i in range(self.num_syllables):
            label_list[i] = self.syllables[i].label
        return(label_list)

if __name__ == "__main__":
    recording_test = Recording('Downloads/CATH1.wav', 'Downloads/CATH1.TextGrid', 'CATH1')
    print(recording_test)
    recording_test.get_annotations(False)
    print(recording_test)
    for i in range(26):
        print(recording_test.syllables[i].start,
          recording_test.syllables[i].end)
