from Syllable import Syllable

class SyllableCollection(object):

    def __init__(self, syllables = []):
        self.__check_inputs(syllables)
        self.syllables = syllables
        self.num_syllables = len(syllables)

    def __str__(self):
        for each in self.syllables:
            print(each)

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
            raise TypeError('Invalid argument type.')

    def __len__(self):
        return self.num_syllables

    def __check_inputs(self, syllables):
        if not isinstance(syllables, list) and not isinstance(syllables, Syllable):
            raise TypeError('<syllables> must be a list of syllables or single syllable instance.')
        if len(syllables) > 0 and not any([self.__check_syllable(each) for each in syllables]):
            raise TypeError('Non-syllable class contained in input list.')

    def __check_syllable(self, syllable):
        if not isinstance(syllable, Syllable):
            return(False)
        return(True)

    def add_syllable(self, new_syllable):
        if not self.__check_syllable(new_syllable):
            raise TypeError('Attempting to set non-Syllable entry.')
        self.syllables.append(new_syllable)
        self.num_syllables += 1

    def get_labels(self):
        label_list = [each.label for each in self.syllables]
        return(label_list)

    def get_start_times(self):
        start_list = [each.start for each in self.syllables]
        return(start_list)

    def get_end_times(self):
        end_list = [each.start for each in self.syllables]
        return(end_list)

if __name__ == '__main__':
    from Recording import Recording
    recording_test = Recording('Downloads/CATH1.wav', 'Downloads/CATH1.TextGrid', 'CATH1')
    print(recording_test)
    recording_test.get_annotations(False)
    print(recording_test)
    b = SyllableCollection(recording_test.syllables[1:5])
    b.add_syllable(recording_test.syllables[6])
    print(b.num_syllables)
    print(b.get_labels())
    print(b.get_start_times())
    print(b.get_end_times())
