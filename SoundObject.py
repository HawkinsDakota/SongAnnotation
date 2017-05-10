import librosa  # sound and music editing package
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from seaborn import distplot
from scipy import signal
from skimage import feature, filters, measure
from copy import deepcopy


class SoundObject(object):

    def __init__(self, start=None, end=None, sound_file=None,
                 preprocess=True, fmin=0, fmax=None):
        self.__check_inputs(start, end, sound_file)
        # provided instance variables
        self.start = start
        self.end = end
        self.sound_file = sound_file
        self.fmin = fmin
        self.fmax = fmax
        # calculated instance variables
        self.preprocess = preprocess
        if sound_file is not None:
            self.sound, self.sample_rate, self.duration = self.get_sound_array()
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
        if start is not None:
            if type(start) is not int and type(start) is not float:
                raise IOError('Expected numeric value for start time.')
        if end is not None:
            if type(end) is not int and type(end) is not float:
                raise IOError('Expected numeric value for end time.')
        if sound_file is not None:
            if type(sound_file) is not str:
                raise IOError('Expected file path for <sound_file>.')

    def get_sound_array(self):
        """
        Return sound data and assocaited variables from `sound_file`.

        Args:
            None.

        Returns:
            tuple (sound, sample_rate, duration):
                sound (:numpy.array:): 1 dimensional numpy array of
                    amplitude values.
                sample_rate (int): rate in hz sound was sampled at.
                duration (float): duration of `sound` in seconds.
        """
        duration = self.end - self.start
        sound, sample_rate = librosa.load(path=self.sound_file,
                                          sr=None,
                                          offset=self.start,
                                          duration=duration)
        return(sound, sample_rate, duration)

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

    def read_sound_file(self):
        """
        Read `sound_file`.

        Reads `sound_file` and instantiates several instance variables
        including `self.sound`, `self.sample_rate`, `self.bin_size`,
        and `self.spectrogram`.

        Args:
            None
        Returns:
            None
        """
        self.sound, self.sample_rate = self.get_sound_array()
        self.bin_size = int(self.sample_rate*0.001)
        if self.preprocess:
            self.__preprocess_sound()
        self.spectrogram = self.__mel_spectrogram()

    def copy(self):
        """Deep copy the current `SoundObject` to a new `SoundObject`."""
        return(deepcopy(self))

    def time_shift(self, shift):
        """
        Shift the begginning or end time of a `SoundObject`.

        Parameters:
            shift (numeric): The value to shift the time frame by. A
                negative value shifts the `SoundObject` backward in time,
                while a positive value shifts the `SoundObject` forward
                in time.

        Returns:
            `SoundObject`: a time-shifted `SoundObject`.
        """
        shifted_sound = SoundObject(start=max([self.start + shift, 0]),
                                    end=max([self.end + shift, 0]),
                                    sound_file=self.sound_file,
                                    preprocess=self.preprocess,
                                    fmin=self.fmin,
                                    fmax=self.fmax)
        return(shifted_sound)

    def canny(self, sigma=1):
        """
        Perform Canny edge detection on a `SoundObject` spectrogram.

        Parameters:
            sigma (numeric): Sigma value to pass to Canny filtering.

        Returns:
            Canny-filtered spectrogram.
        """
        edges = feature.canny(self.spectrogram, sigma=sigma)
        return(self.spectrogram*edges)

    def normalize(self):
        """Z normalize a `SoundObject` spectrogram."""
        spec_avg = np.mean(self.spectrogram)
        spec_dev = np.std(self.spectrogram)
        self.spectrogram = (self.spectrogram - spec_avg)/spec_dev

    def scale(self, smin=0, smax=1):
        """
        Scale `SoundObject` spectrogram.

        Scales a `SoundObject` spectrogram between `min` and `max`.

        Parameters:
            smin (numeric, optional): Lower boundary of scaled spectrogram.
                Default is 0.

            smax (numeric, optional): Upper boundary of scaled spectrogram.
                Default is 1.
        """
        spec_min = np.min(self.spectrogram)
        spec_max = np.max(self.spectrogram)
        scale_quotient = (smax - smin)/(spec_max - spec_min)
        scaled = scale_quotient*(self.spectrogram - spec_min) + smin
        return(scaled)

    def filter(self, method='Scharr'):
        """
        Perform image filtering on a `SoundObject` spectrogram.

        Performs either Scharr, Sobel, Roberts, or Prewitt image filtering
        on a `SoundObject` spectrogram.

        Parameters:
            method (str, optional): Filtering method. Accepted values are
                either Scharr, Sobel, Roberts, Prewitt, Hessian.
                Default method is Scharr.

        Returns:
            (:numpy:ndarray): filtered `SoundObject` spectrogram.
        """
        filtered = self.spectrogram
        method = method.upper()
        if method == 'SCHARR':
            filtered = filters.scharr(self.spectrogram)
        elif method == 'SOBEL':
            filtered = filters.sobel(self.spectrogram)
        elif method == 'ROBERTS':
            filtered = filters.roberts(self.spectrogram)
        elif method == 'PREWITT':
            filtered = filters.prewitt(self.spectrogram)
        elif method == 'HESSIAN':
            filtered = filters.hessian(self.spectrogram)
        elif method == 'GABOR':
            filtered = filters.gabor(self.spectrogram,
                                     frequency=0.5)[0]
        return(filtered)

    def binary_mask(self, threshold=2):
        """
        Lay a binary mask over a `SoundObject` spectrogram.

        Performs binary filtering over a `SoundObject` spectrogram. Values
        below the threshold value will be set to 0, while values above the
        threshold are set to 1.

        Args:
            threshold (numeric, optional): threshold value to determine
                filtered values. Default value is 2.
        Returns:
            (:numpy.ndarray:) filtered `SoundObject` spectrogram.
        """
        adj_matrix = self.spectrogram
        adj_matrix[adj_matrix < 2] = 0
        adj_matrix[adj_matrix >= 2] = 1
        return(adj_matrix)

    def contours(self, start_value=2, end_value=None, step=0.1):
        """
        Find contours within images.

        Finds contours within a `SoundObject` spectrogram along a
        spectrum of values. Contours are found for values within the
        interval [`start_value`, `end_value`].

        Args:
            start_value (numeric, optional): Lower boundary for contour
                detection. Default value is 2.
            end_value (numeric, optional): Upper boundary for contour
                detection. Default is None. If left as None the upper
                boundary will be set to the maximum value within the
                `SoundObject` spectrogram.
            step (numeric, optional): step size to increase by for each
                pass at contour detection. Default is 0.1.
        Returns:
            (:numpy.ndarray:, int): Adjacenty matrix of zeros and ones
                where 1 denotes detected contours.
        """
        contour_matrix = np.zeros(test.spectrogram.shape)
        rounded = np.around(test.spectrogram, 1)
        if end_value is None:
            end_value = np.max(rounded)

        iter_values = np.arange(start_value, end_value + step, step)
        if start_value == end_value:
            iter_values = [start_value]

        for value in iter_values:
            contours = measure.find_contours(test.spectrogram, value)
            if len(contours) > 0:
                for i, contour in enumerate(contours):
                    row_idx = [int(each) for each in contour[:, 0]]
                    col_idx = [int(each) for each in contour[:, 1]]
                    contour_matrix[row_idx, col_idx] = 1
        return(contour_matrix)

    def add_noise(self, mean=0, std=1):
        """
        Add Gaussian noise to a `SoundObject` spectrogram.

        Parameters:
            mean (numeric, optional): Mean of the Gaussian noise. Default
                is zero.
            std (numeric, optional): Standard deviation of the Gaussian
                noise. Default is zero.
        Returns:
            numpy.ndarray: A spectrogram with added noise.
        """
        noise = np.random.normal(mean, std, self.spectrogram.shape)
        return(self.spectrogram + noise)

    def plot(self, show=False, cmap='viridis'):
        """
        Plot a spectrogram of `SoundObject`.

        Args:
            show (boolean, optional): whether to show plotted spectrogram.
                Default is False.
            cmap (str, optional): color map to use for plotting. Default
                is viridis.
        Returns:
            (:matplotlib.figure:) object for the plotted spectrogram.
        """
        #  time_breaks = np.array([0]*self.spectrogram.shape[1])
        #  for i in range(self.spectrogram.shape[1]):
        #      time_breaks[i] = self.bin_size/self.sample_rate*i + self.start

        fig, ax = plt.subplots()
        plot_features = {'cmap': cmap}
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
        return(fig)

    def plot_distribution(self):
        amplitude_array = self.spectrogram.flatten()
        ax = distplot(amplitude_array,
                      rug=True,
                      kde=False)
        plt.savefig('dist_test.png')
        return(ax)


if __name__ == '__main__':

    test = SoundObject(start=11.844455261385376,
                       end=12.081455063757392,
                       sound_file='Downloads/CATH1.WAV',
                       preprocess=False,
                       fmin=500,
                       fmax=11000)
    test.normalize()
    test1 = test.copy()
    test2 = test.copy()
    test_contours = test1.contours()
    test1.spectrogram = test_contours
    test2.spectrogram = test2.binary_mask()

    test.plot(True)
    test1.plot(True, 'Greys')
    test2.plot(True, 'Greys')
    #
    # process_test = SoundObject(start=11.844455261385376,
    #                            end=12.081455063757392,
    #                            sound_file='Downloads/CATH1.WAV',
    #                            preprocess=False)
    # print(test == process_test)

    # big_test = SoundObject(start = 13,
    #                        end = 20,
    #                        sound_file = 'Downloads/CATH1.WAV',
    #                        preprocess = True,
    #                        fmin = 1000,
    #                        fmax = 10000)
    # print(test == big_test)
    #
    # big_test_preprocess = SoundObject(start = 13,
    #                                   end = 20,
    #                                   sound_file = 'Downloads/CATH1.WAV',
    #                                   preprocess = False)
    # big_test_preprocess.plot_spectrogram(True)
