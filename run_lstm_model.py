import SoundDataSet
from SyllableCollection import SyllableCollection
from SoundDataSet import SoundDataSet
from SyllableLSTM import SyllableLSTM
import SyllablePlots
import pickle
import pandas
import numpy
import torch
from sklearn.model_selection import train_test_split

def load_dataset(read_file):
    with open(read_file, 'rb') as pickle_input:
        return(pickle.load(pickle_input))

def create_training_and_test(syllables, fold=10):
    test_size = int(len(syllables)/fold)
    training_size = len(syllables) - test_size
    all_indices = range(len(syllables))
    training, test = train_test_split(all_indices,
                                      test_size=test_size,
                                      train_size=training_size,
                                      stratify=syllables.get_labels())
    return(training, test)

def calculate_weights(syllables):
    unique, counts = numpy.unique(syllables, return_counts=True)
    weights = counts/sum(counts)
    return(list(unique), torch.from_numpy(weights).float())

#
# dataset = load_dataset('all_songs.pkl')
# # SyllablePlots.plot_syllable_distribution(dataset.syllable_labels,
# #                                          save_file='all_songs_dist.csv',
# #                                          plot_file='all_songs_dist.png')
# count_df = pandas.read_csv('all_songs_dist.csv')
# keep_labels = count_df['Syllable'].loc[count_df['Count'] > 5].values
#
# keep_syllables = SyllableCollection([each for each in dataset.syllables if each.label in keep_labels])
#
# # SyllablePlots.plot_syllable_distribution(keep_syllables.get_labels(),
# #                                          save_file='subset_syl_dist.csv',
# #                                          plot_file='subset_syl_dist.png')
#
# training, test = create_training_and_test(keep_syllables)
# unique_syllables, training_weights = calculate_weights(keep_syllables.get_labels())
# syllable_lstm = SyllableLSTM(128, unique_syllables, training_weights)
# syllable_lstm.train_model(keep_syllables[training],
#                           n_epochs=500,
#                           save_file='weighted_lstm.pkl',
#                           loss_file='weighted.csv',
#                           track_time=True)
# c_matrix = syllable_lstm.test_model(keep_syllables[test])
# c_matrix.to_csv('weighted_cmatrix.csv')
# SyllablePlots.plot_confusion(c_matrix, 'weighted_cmatrix.png')

single_syllable = load_dataset('single_syllable.pkl')
one_of_each = SyllableLSTM(128,
                           single_syllable.syllables.get_unique_syllables(),
                           50)
one_of_each.train_model(single_syllable.syllables,
                        n_epochs=500,
                        save_file='single_syllable_lstm.pkl',
                        loss_file='single_syllable_loss.csv',
                        track_time=False)
