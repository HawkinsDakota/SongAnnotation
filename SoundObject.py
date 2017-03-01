import librosa #sound and music editing package
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class SoundObject(object):

    def __init__(self, start, end, sound_file):
        self.__check_inputs(start, end, sound_file)
        # provided instance variables
        self.start = start
        self.end = end
        self.duration = end - start
        self.sound_file = sound_file
        # calculated instance variables
        self.sound, self.sample_rate = self.__get_sound_array()
        self.spectrogram = self.__generate_spectrogram()

    def __str__(self):
        out = '''
        Start: %0.2f
        End: %0.2f
        Duration: %0.2f''' % (self.start, self.end, self.duration,)
        return(out)

    def __check_inputs(self, start, end, sound_file):
        if type(start) != int and type(start) != float:
            raise IOError('Expected numeric value for start time.')

        if type(end) != int and type(end) != float:
            raise IOError('Expected numeric value for end time.')

        if type(sound_file) != str:
            raise IOError('Expected file path for <sound_file>.')

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
        plt.figure()
        plot_features = {'cmap' : 'viridis'}
        librosa.display.specshow(self.spectrogram,
                                 sr = self.sample_rate,
                                 x_axis = 'time',
                                 y_axis = 'mel',
                                 **plot_features)
        plt.colorbar(format = '%+0.2f dB')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    test = SoundObject(start = 11.844455261385376,
                    end = 12.081455063757392,
                    sound_file = 'Downloads/CATH1.WAV')

    print(test)
#test.plot_spectrogram()

#big_test = SoundObject(start = 0,
#                end = 15,
#                sound_file = 'Downloads/CATH1.WAV')
#big_test.plot_spectrogram()
