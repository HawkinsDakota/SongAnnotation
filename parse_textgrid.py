import re
from collections import namedtuple

def __check_inputs(input_file, species_id):
    if type(input_file) != str:
        raise IOError('Expected file path for <input_file>.')

    if type(species_id) != str:
        raise IOError('Expected string for <species_id>.')

    if not input_file.endswith('.TextGrid'):
        raise IOError('''File does not end with .TextGrid. 
        File must be in .TextGrid format.''')

def get_annotations(input_file, species_id):
    '''
    Parses a .TextGrid file for syllable annotations
    '''
    __check_inputs(input_file, species_id)
    syllables = {}
    SoundSection = namedtuple('SoundSection', ['start', 'end', 'label'])
    num_syllables = 1
    with open(input_file) as read_file:
        for i, line in enumerate(read_file):
            if re.search('intervals \[[0-9]*\]', line) != None:
                start_line = read_file.readline().strip()
                end_line = read_file.readline().strip()
                label_line = read_file.readline().strip()
                label = re.sub(r'^"|"$', '', label_line.split('=')[1].strip())
                if label != '':
                    start = float(start_line.split('=')[1])
                    end = float(end_line.split('=')[1])
                    label = species_id + '_' + label
                    syllables[num_syllables] = SoundSection(start, end, label)
                    num_syllables += 1
    return(syllables)

def get_silence(input_file, species_id):
    return(None)
    
def get_syllables(input_file, species_id):
    return(None)
    
x = get_syllables('Downloads/CATH1.TextGrid', 'CATH')
test_syllable = x[1]
print(test_syllable)
