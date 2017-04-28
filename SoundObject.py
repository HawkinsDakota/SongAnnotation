import librosa  #sound and music editing package
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class SoundObject(object):

    def __init__(self, start, end, sound_file=None,
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
        if sound_file is not None:
            self.sound, self.sample_rate = self.__get_sound_array(preprocess)
            if preprocess:
                self.__preprocess_sound()
            # bin sound into 1 ms windows
            self.bin_size = int(self.sample_rate*0.001)
            self.spectrogram = self.__mel_spectrogram()

    def __str__(self):
        out = '''
        Start: %0.2f
        End: %0.2f
        Duration: %0.2f''' % (self.start, self.end, self.duration,)
        return(out)

    def __eq__(self, other):
        start_eq = self.start == other.start
        end_eq = self.end == other.end
        sound_eq = self.sound_file == other.sound_file
        return(all([start_eq, end_eq, sound_eq]))

    def __check_inputs(self, start, end, sound_file):
        if type(start) != int and type(start) != float:
            raise IOError('Expected numeric value for start time.')

        if type(end) != int and type(end) != float:
            raise IOError('Expected numeric value for end time.')

        if type(sound_file) != str and sound_file is not None:
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

    def __mel_spectrogram(self):

        mel_spec = librosa.feature.melspectrogram(self.sound,
                                                  sr=self.sample_rate,
                                                  hop_length=self.bin_size,
                                                  n_mels=256,
                                                  fmin=self.fmin,
                                                  fmax=self.fmax)
        log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
        return(log_mel_spec)

    def plot(self):
        Pxx, freqs, bins, im = plt.specgram(self.sound,
                                            NFFT=256,
                                            Fs=self.sample_rate)
        plt.show()

        # decibel = librosa.amplitude_to_db(self.sound, ref=max)
        # spectrum, freqs, t, im = signal.spectrogram(decibel,
        #                                        fs=self.sample_rate)
        # plt.pcolormesh(t, f, spectrogram, cmap='viridis')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

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
    #test.plot()
    #
    # process_test = SoundObject(start=11.844455261385376,
    #                            end=12.081455063757392,
    #                            sound_file='Downloads/CATH1.WAV',
    #                            preprocess=False)
    #print(test == process_test)

    # big_test = SoundObject(start = 13,
    #                        end = 20,
    #                        sound_file = 'Downloads/CATH1.WAV',
    #                        preprocess = True,
    #                        fmin = 1000,
    #                        fmax = 10000)
    #print(test == big_test)
    #
    # big_test_preprocess = SoundObject(start = 13,
    #                                   end = 20,
    #                                   sound_file = 'Downloads/CATH1.WAV',
    #                                   preprocess = False)
    # big_test_preprocess.plot_spectrogram(True)
