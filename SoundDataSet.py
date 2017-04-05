from Recording import Recording
import glob
import re
import os

class SoundDataSet(object):

    def __init__(self, data_directory, species='Unassigned', preprocess=True,
                 fmin=0, fmax=None):
        self.wav_to_grid = match_wav_with_textgrid(data_directory)
        self.syllables, self.unique_syllables = self.__load_syllables(
            species, preprocess, fmin, fmax)
        self.n_syllables = len(self.syllables)

    def match_wav_with_textgrid(data_directory):
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

    def __load_syllables(self, species, preprocess, fmin, fmax):
        syllables = []
        unique_syllables = set()
        for wav in self.wav_to_grid:
            new_recording = (Recording(sound_file=wav,
                                       grid_file=self.wav_to_grid[wav],
                                       species=species,
                                       preprocess=preprocess,
                                       fmin=fmin,
                                       fmax=fmax))
            new_recording.get_annotations()
            syllables += new_recording.syllables
            unique_syllables.union(new_recording.list_syllables())
        return(syllables, list(unique_syllables))

    def _create_training_and_test(fold=10):

        return (None)
