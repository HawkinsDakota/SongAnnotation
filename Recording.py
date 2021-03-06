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
        # must specifically instantiate syllables as empty -- for some reason
        self.syllables = SyllableCollection([])
        self.n_syllables = 0
        self.songs = []
        self.num_songs = 0
        self.background = []
        self.num_background = 0
        self.__syllable_dictionary = self.__load_dictionary('syllable_dictionary.csv')


    def __load_dictionary(self, dictionary_file):
        with open(dictionary_file) as f:
            lines = f.readlines()
        return([x.strip().split(',')[0] for x in lines])

    def __str__(self):
        out = '''
Species: %s
Audio file: %s
TextGrid file: %s
Number of syllables: %i
Number of songs: %i
Total number of annotations: %i
              ''' % (self.species, self.sound_file,
                     self.grid_file, self.n_syllables, self.num_songs,
                     self.n_syllables + self.num_background)
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
        new_syllable = Syllable(start, end, self.sound_file,
                                self.species, label)
        if new_syllable not in self.syllables.to_list():
            self.syllables.add_syllable(new_syllable)
            self.n_syllables += 1

    def write_new_syllable():
        return(0)

    def new_background(self, start, end):
        new_background = Syllable(start, end, self.sound_file,
                                  species = 'None', label = 'background')
        self.background.append(new_background)
        self.num_background += 1

    def get_annotations(self, keep_background=True, keep_songs=False):
        '''
        Parses a .TextGrid file for syllable annotations
        '''
        read_file = open(self.grid_file, 'r')
        lines = read_file.readlines()
        read_file.close()
        n_lines = len(lines)
        current_line = 0
        bar = ProgressBar(max_value=n_lines)
        while current_line < n_lines:
            bar.update(current_line + 1)
            line = lines[current_line]
            if re.search('intervals \[[0-9]*\]', line) is not None:
                start_line = lines[current_line + 1].strip()
                end_line = lines[current_line + 2].strip()
                label_line = lines[current_line + 3].strip()
                start = float(start_line.split('=')[1])
                end = float(end_line.split('=')[1])
                label = re.sub(r'^"|"$', '',
                               label_line.split('=')[1].strip())
                if label in self.__syllable_dictionary:
                    label = self.species + '_' + label
                    self.new_syllable(start, end, label)
                current_line += 4
            else:
                current_line += 1
        bar.update(n_lines)
        # with open(self.grid_file) as read_file:
        #     print('Parsing %s and %s' % (self.grid_file, self.sound_file))
        #     bar = ProgressBar(max_value=UnknownLength)
        #     song_start = 0
        #     for i, line in enumerate(read_file):
        #         bar.update(i)
        #         if re.search('intervals \[[0-9]*\]', line) is not None:
        #             start_line = read_file.readline().strip()
        #             end_line = read_file.readline().strip()
        #             label_line = read_file.readline().strip()
        #             start = float(start_line.split('=')[1])
        #             end = float(end_line.split('=')[1])
        #             label = re.sub(r'^"|"$', '',
        #                            label_line.split('=')[1].strip())
        #             if label != '':
        #                 label = self.species + '_' + label
        #                 self.new_syllable(start, end, label)
                        # if keep_songs:
                        #     current_syllable = self.syllables[self.n_syllables - 1]
                        #     past_syllable = self.syllables[self.n_syllables - 2]
                        #     if (self.n_syllables > 1 and (current_syllable.end - past_syllable.start > 1)):
                        #         self.add_song(self.syllables[song_start:(self.n_syllables - 1)],
                        #                       self.syllables[song_start].start,
                        #                       current_syllable.end,
                        #                       self.sound_file,
                        #                       self.preprocess,
                        #                       self.fmin,
                        #                       self.fmax)
                        #         song_start = self.n_syllables
                    # elif keep_background:
                    #     self.new_background(start, end)
            # if keep_songs:
            #     self.add_song(self.syllables[song_start:(self.n_syllables - 1)],
            #                   self.syllables[song_start].start,
            #                   self.syllables[self.n_syllables - 1].end,
            #                   self.sound_file,
            #                   self.preprocess,
            #                   self.fmin,
            #                   self.fmax)

    def add_song(self, song_syllables, song_start, song_end, sound_file,
                 preprocess, fmin, fmax):
        self.songs.append(Song(song_syllables, song_start, song_end,
                               self.sound_file, self.preprocess,
                               self.fmin, self.fmax))
        self.num_songs += 1

    def syllable_lables(self):
        return(self.syllables.get_labels())

    def unique_syllable_labels(self):
        return(self.syllables.get_unique_syllables())

    def check_for_unique_syllables(self):
        same_syllables = []
        for i, each in enumerate(self.syllables.to_list()):
            for j in range(i+1, len(self.syllables.to_list()) - 1):
                if self.syllables[i] == self.syllables[j]:
                    same_syllables.append(self.syllables)
        return(len(same_syllables) == 0 )


if __name__ == "__main__":
    recording_test = Recording('Downloads/CATH2.wav',
                               'TestData/CATH2.TextGrid',
                               'CATH')
    # 24 entries
    recording_test.get_annotations(keep_background=False)
    print(recording_test)
    print(len(recording_test.unique_syllable_labels()))
    print(recording_test.check_for_unique_syllables())
