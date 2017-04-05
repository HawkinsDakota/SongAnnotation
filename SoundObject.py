import librosa  #sound and music editing package
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class SoundObject(object):

    def __init__(self, start, end, sound_file,
      preprocess = True, fmin = 0, fmax = None):
        self.__check_inputs(start, end, sound_file)
        # provided instance variables
        self.start = start
        self.end = end
        self.duration = end - start
        self.sound_file = sound_file
        self.fmin = fmin
        self.fmax = fmax
        # calculated instance variables
        self.sound, self.sample_rate = self.__get_sound_array(preprocess)
        if preprocess:
            self.__preprocess_sound()
        self.bin_size = int(self.sample_rate*0.001)  # bin sound into 1 ms windows
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

    def __get_sound_array(self, preprocess):
        sound, sample_rate = librosa.load(path=self.sound_file,
                                          sr=None,
                                          offset=self.start,
                                          duration=self.duration)
        return(sound, sample_rate)

    def __preprocess_sound(self):

        # run pre-emphasis filtering to increase signal to noise ratio
        filter_coefficient = 0.95  # alpha = 0.95 or 0.97
        pre_emph_signal = np.append(self.sound[0],
                                    self.sound[1:] -
                                    filter_coefficient * self.sound[:-1])
        self.sound = pre_emph_signal

    def __generate_spectrogram(self):

        mel_spec = librosa.feature.melspectrogram(self.sound,
                                                  sr=self.sample_rate,
                                                  hop_length=self.bin_size,
                                                  n_mels=128,
                                                  fmin=self.fmin,
                                                  fmax=self.fmax)
        log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
        return(log_mel_spec)

    def plot_spectrogram(self, show=False):
        #  time_breaks = np.array([0]*self.spectrogram.shape[1])
        #  for i in range(self.spectrogram.shape[1]):
        #      time_breaks[i] = self.bin_size/self.sample_rate*i + self.start

        plt.figure()
        plot_features = {'cmap': 'viridis'}
        librosa.display.specshow(self.spectrogram,
                                 sr=self.sample_rate,
                                 x_axis='time',
                                 y_axis='mel',
                                 hop_length=self.bin_size,
                                 fmin=self.fmin,
                                 fmax=self.fmax,
                                 **plot_features)
        plt.colorbar(format='%+0.2f dB')
        plt.tight_layout()
        if show:
            plt.show()

if __name__ == '__main__':
    test = SoundObject(start=11.844455261385376,
                       end=12.081455063757392,
                       sound_file='Downloads/CATH1.WAV',
                       preprocess=True)
    test.plot_spectrogram(True)

    # process_test = SoundObject(start = 11.844455261385376,
    #                            end = 12.081455063757392,
    #                            sound_file = 'Downloads/CATH1.WAV',
    #                            preprocess = False)
    # test.plot_spectrogram(True)

    # big_test = SoundObject(start = 13,
    #                        end = 20,
    #                        sound_file = 'Downloads/CATH1.WAV',
    #                        preprocess = True,
    #                        fmin = 1000,
    #                        fmax = 10000)
    # big_test.plot_spectrogram(True)
    #
    # big_test_preprocess = SoundObject(start = 13,
    #                                   end = 20,
    #                                   sound_file = 'Downloads/CATH1.WAV',
    #                                   preprocess = False)
    # big_test_preprocess.plot_spectrogram(True)
