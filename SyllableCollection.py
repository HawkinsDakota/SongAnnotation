from Syllable import Syllable

class SyllableCollection(object):

    def __init__(self, syllables = []):
        self.__check_inputs(syllables)
        self.syllables = syllables
        self.n_syllables = len(syllables)

    def __str__(self):
        out_str = ''
        for each in self.syllables:
            out_str += str(each)
        return(out_str)

    def __add__(self, other):
        if isinstance(other, SyllableCollection):
            return(SyllableCollection(self.syllables + other.syllables))
        else:
            raise TypeError('Cannot concatenate \
            SyllableCollection with %s' % type(other))

    def __iadd__(self, other):
        if isinstance(other, SyllableCollection):
            self.syllables = self.syllables + other.syllables
            self.n_syllables = len(self.syllables)
            return(self)
        else:
            raise TypeError('Cannot concatenate \
            SyllableCollection with %s' % type(other))

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.stop >= len(self) and i.start >= len(self):
                raise IndexError('The index (%d) is out of range' % i)
            return([self[j] for j in range(*i.indices(len(self)))])

        elif isinstance(i, int):
            if i < 0:
                i += len(self)
            if i >= len(self):
                raise IndexError('The index (%d) is out of range' % i)
            return(self.syllables[i])

        else:
            raise TypeError('Invalid argument type: %s.' % type(i))

    def __len__(self):
        return(self.n_syllables)

    def __check_inputs(self, syllables):
        if not isinstance(syllables, list):
            raise TypeError('<syllables> must be a list of syllables.')
        if len(syllables) > 0 and not any([self.__check_syllable(each) for each in syllables]):
            raise TypeError('Non-syllable class contained in input list.')

    def __check_syllable(self, syllable):
        if not isinstance(syllable, Syllable):
            return(False)
        return(True)

    def add_syllable(self, new_syllable):
        if not self.__check_syllable(new_syllable):
            raise TypeError('Attempting to set non-Syllable entry: %s' %
                             type(new_syllable))
        self.syllables.append(new_syllable)
        self.n_syllables += 1

    def get_labels(self):
        label_list = [each.label for each in self.syllables]
        return(label_list)

    def get_unique_syllables(self):
        return(list(set(self.get_labels())))

    def get_start_times(self):
        start_list = [each.start for each in self.syllables]
        return(start_list)

    def get_end_times(self):
        end_list = [each.start for each in self.syllables]
        return(end_list)


if __name__ == '__main__':
    syl_1 = Syllable(start=12.19, end=12.43, sound_file='TestData/CATH1.wav')
    syl_2 = Syllable(start=12.60, end=12.85, sound_file='TestData/CATH1.wav')
    syl_3 = Syllable(start=13.03, end=13.19, sound_file='TestData/CATH1.wav')
    syl_col_test = SyllableCollection([syl_1, syl_2, syl_3])
