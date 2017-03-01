import librosa #sound and music editing package
import numpy as np
import matplotlib.pyplot as plt


class Syllable(object):

    def __init__(self, start, end, sound_file, label = None, species = None):
        # provided instance variables
        self.start = start
        self.end = end
        self.duration = end - start
        self.sound_file = sound_file
        self.species = species
        self.label = label
        # calculated instance variables
        self.sound, self.sample_rate = self.__get_sound_array()
        self.spectrogram = self.__generate_spectrogram()

    def __get_sound_array(self):
        filter_coefficient = 0.95 # alpha = 0.95 or 0.97
        sound, sample_rate = librosa.load(path = self.sound_file, 
                                          sr = None, 
                                          offset = self.start, 
                                          duration = self.duration)
        pre_emph_signal = np.append(sound[0], 
                                    sound[1:] - filter_coefficient * sound[:-1])
        return(pre_emph_signal, sample_rate)
        
    def __generate_spectrogram(self):
        length_of_1_ms_windows = int(self.sample_rate*0.001)
        mel_spec = librosa.feature.melspectrogram(self.sound,
                                            sr = self.sample_rate,
                                            hop_length = length_of_1_ms_windows,
                                            n_mels = 250)
        log_mel = librosa.logamplitude(mel_spec, ref_power = np.max)
        return(log_mel)
        
    def plot_spectrogram(self):
        plt.figure(figsize = (12,4))
        plot_features = {'cmap' : 'viridis'}
        librosa.display.specshow(self.spectrogram,
                                 sr = self.sample_rate,
                                 x_axis = 'time', 
                                 y_axis = 'mel',
                                 **plot_features)
        plt.colorbar(format = '%+0.2f dB')
        if self.label is None:
            syl = 'Unknown Syllable'
        else:
            syl = self.label
        plt.title('Mel-Power Spectrogram for %s' %syl)
        
        plt.tight_layout()
        
        
test = Syllable(start = 11.844455261385376,
                end = 12.081455063757392,
                sound_file = 'Downloads/CATH1.WAV',
                label = 'CATH_nay',
                species = 'CATH')

test.plot_spectrogram()

big_test = Syllable(start = 0,
                end = 15,
                sound_file = 'Downloads/CATH1.WAV',
                label = 'CATH_nay',
                species = 'CATH')
big_test.plot_spectrogram()