"""
Transform Syllables.

Author: Dakota Hawkins
Wednesday May 10, 2017
"""
import numpy as np
from collections import namedtuple
from os import path
from matplotlib import pyplot as plt
from Syllable import Syllable
from SoundObject import normalize_spectrogram

def transform(syllable, noise_params, prefix='', edge_threshold=2):
    """
    Perform binary masking and noise simulation for a given `Syllable`.

    Args:
        syllable (Syllable): a time and frequency constant syllable.
        noise_params (list, namedtuple): list of tuples containing mean
            and standard deviation to use when simulating noise. (mean, std).
        prefix (str, optional): string to append to beginning of transformation
            labels. Default is empty.
        edge_threshold (float, optional): threshold to use for binary masking.
            Default is 2.
    Returns:
        (list, :numpy.ndarray:): transformed spectrograms.
        (list, str): labels denoting performed transformations
    """
    transformed = [1 + len(noise_params)]
    labels = [1 + len(noise_params)]
    transformed[0] = syllable.binary_mask(edge_threshold)
    labels[0] = "{0}bm".format(edge_threshold)
    for i, params in enumerate(noise_params):
        noisey = syllable.add_noise(params[0], params[1])
        transformed[i + 1] = normalize_spectrogram(noisey)
        labels[i + 1] = "{0}-{1}n".format(params[0], params[1])
    return((transformed, labels))


def transformation_id(parent_syllable, tx_string):
    """
    Generate unique id for transformed syllable.

    Generates a unique id for a transformed syllable by taking meta information
    from the parent syllable, and combining it with a string representation of
    the performed transformations.

    Args:
        parent_syllable (Syllable): original syllable object.
        tx_string (str): string representing the conducnted transformations.
    Returns:
        (str): unique id for transformed spectrogram.
    """
    parent_id = parent_syllable.generate_id()
    return(parent_id + '_' + tx_string)


def apply_transformations(syllable):
    """
    Apply a standardized set of transformations to a `Syllable` object.

    Args:
        syllable (Syllable): syllable to apply transformations to.
    Returns:
        (list, Syllable): list of transformed spectrograms.
    """
    TIME_SHIFTS = np.around(np.arange(-0.1, 0.11, 0.01), 2)
    FREQ_SHIFTS = np.arange(-5, 6, 1)
    noise = namedtuple('noise', ['mean', 'std'])
    NOISE_PARAMS = [noise(0, 1), noise(0, 1.5), noise(0, 2)]
    transformed = []
    tx_labels = []

    for t in TIME_SHIFTS:
        t_shifted = syllable.time_shift(t)
        t_label = str(t) + 't'
        if t != 0:
            transformed.append(t_shifted)
            tx_labels.append(syllable.generate_id + '_' + t_label)
        for f in FREQ_SHIFTS:
            ft_shifted = t_shifted.freq_shift(f)
            ft_label = t_label + '_' + str(f) + 'f'
            if f != 0:
                transformed.append(ft_shifted)
                tx_labels.append(syllable.generate_id + '_' + ft_label)
            for each in NOISE_PARAMS:
                spectrograms, labels = transform(ft_shifted,
                                                 each,
                                                 prefix=ft_label)
                transformed += spectrograms
                tx_labels + labels
    # this will change so looping only occurs once -- testing purposes.
    temp = syllable.copy()
    for i, each in enumerate(transformed):
        temp.spectrogram = each
        fig = temp.plot()
        plt.savefig('Transformations/' + tx_labels[i])
        plt.close()

if __name__ == '__main__':
    test = Syllable(start=11.844455261385376,
                    end=12.081455063757392,
                    sound_file='Downloads/CATH1.WAV',
                    species='CATH',
                    label='yea')
    apply_transformations(test)
