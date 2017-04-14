from Recording import Recording
import glob
from SyllableCollection import SyllableCollection
import re
import os
import pickle
from numpy import random


class SoundDataSet(object):

    def __init__(self, data_directory, species='Unassigned', preprocess=True,
                 fmin=0, fmax=None):
        # provided instance variables
        self.species = species
        self.__preprocess = preprocess
        self.__fmin = fmin
        self.__fmax = fmax

        # calculated instance variables
        self.wav_to_grid = self.match_wav_with_textgrid(data_directory)
        # must specifically instantiate syllables as empty -- for some reason
        self.syllables = SyllableCollection([])
        self.syllable_labels = []
        self.unique_syllables = []
        self.n_syllables = 0

    def __len__(self):
        return(len(self.syllables))

    def match_wav_with_textgrid(self, data_directory):
        wav_files = [wav for wav in glob.glob(data_directory + '/*.wav')]
        grid_files = [grid for grid in glob.glob(data_directory +
                                                 '/*.TextGrid')]
        wav_to_grid = {}

        for each in wav_files:
            file_pattern = os.path.basename(each).split('.')[0]
            grid_matches = []
            for grid in grid_files:
                if re.search(file_pattern, grid) is not None:
                    grid_matches.append(grid)
            if len(grid_matches) == 0:
                print('No easy matches found for %s' % each)
            elif len(grid_matches) == 1:
                wav_to_grid[each] = grid_matches[0]
            else:
                print('Multiple matches for %s: %s' %
                      (each, ', '.join(grid_matches)))
        return(wav_to_grid)

    def add_wav_to_grid(self, wav_file, grid_file):
        if not wav_file.endswith('.wav'):
            raise IOError('%s must be a .wav file.' % wav_file)
        if not grid_file.endswith('.TextGrid'):
            raise IOError('%s but a .TextGrid file.' % grid_file)
        if os.path.isfile(wav_file) and os.path.isfile(grid_file):
            self.wav_to_grid[wav_file] = grid_file
        elif not os.path.isfile(wav_file):
            raise IOError('%s file does not exist.' % wav_file)
        else:
            raise IOError('%s file does not exist.' % grid_file)

    def load_syllables(self, save_file=None):
        for wav in self.wav_to_grid:
            print('Loading annotations from: {0}'.format(wav))
            new_recording = (Recording(sound_file=wav,
                                       grid_file=self.wav_to_grid[wav],
                                       species=self.species,
                                       preprocess=self.__preprocess,
                                       fmin=self.__fmin,
                                       fmax=self.__fmax))
            new_recording.get_annotations(keep_background=False,
                                          keep_songs=False)
            self.syllables = self.syllables.combine_collections(new_recording.syllables)
            self.syllable_labels += new_recording.syllable_lables()
        self.unique_syllables = list(set(self.syllable_labels))
        self.n_syllables = len(self.syllables)
        if save_file is not None:
            self.save_dataset(save_file)

    def save_dataset(self, save_file):
        with open(save_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def create_training_and_test(self, fold=10):
        test_size = int(len(self.syllables)/fold)
        all_indices = range(self.n_syllables)
        test_indices = random.choice(all_indices,
                                     size=test_size, replace=False)
        training_indices = list(set(all_indices).difference(test_indices))
        return(list(training_indices), list(test_indices))


def load_dataset(read_file):
    with open(read_file, 'rb') as pickle_input:
        return(pickle.load(pickle_input))

if __name__ == '__main__':
    all_syllables = SoundDataSet('Downloads/', species='CATH')
    all_syllables.add_wav_to_grid('Downloads/SMMJAN13TRK1.wav',
                                  'Downloads/SMMJAN13TRK1.TextGrid')
    all_syllables.add_wav_to_grid('Downloads/SMMJAN13TRK10.wav',
                                  'Downloads/SMMJAN13TRK10.TextGrid')
    all_syllables.add_wav_to_grid('Downloads/TRKa51-3-12.wav',
                                  'Downloads/TRKa51a-3-12.TextGrid')
    all_syllables.add_wav_to_grid('Downloads/TRKa51-3-12.wav',
                                  'Downloads/TRKa51b-3-12.TextGrid')
    all_syllables.add_wav_to_grid('Downloads/TRKb13-3-12.wav',
                                  'Downloads/TRKb13a-3-12.TextGrid')
    all_syllables.add_wav_to_grid('Downloads/TRKb13-3-12.wav',
                                  'Downloads/TRKb13b-3-12.TextGrid')
    all_syllables.load_syllables(save_file='all_songs.pkl')
