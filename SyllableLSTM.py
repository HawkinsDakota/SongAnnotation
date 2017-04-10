import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

import pickle

import SoundDataSet

torch.manual_seed(1)

class SyllableLSTM(nn.Module):

    def __init__(self, input_dim, possible_labels):
        super(SyllableLSTM, self).__init__()
        # input dim should be the number of mels in a syllable slice
        self.input_dim = input_dim
        # output dim will be the number of possible labels
        self.output_dim = len(possible_labels)
        self.labels = possible_labels
        self.label_index_dict = self.create_label_dictionary()
        # hidden dim will be output_dim
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.input_dim, self.output_dim)
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

    # Initialize hidden state of model, includes cell state and
    # hidden state as tuple
    def init_hidden(self):
        return(autograd.Variable(torch.zeros(1, 1, self.output_dim)),
               autograd.Variable(torch.zeros(1, 1, self.output_dim)))

    # Run forward pass on syllable slice over model:
    #  LSTM -> log softmax -> scores
    def forward(self, syllable_slice):
        lstm_out, self.hidden = self.lstm(syllable_slice, 1, -1)
        syllable_scores = F.log_softmax(lstm_out)
        return(syllable_scores)

    # Calcuate score for a given syllable, runs over all
    # slices of syllabe spectrogram
    def calculate_syllable_score(self, syllable):
        syl_spectrogram = syllable.spectrogram
        for j in range(syl_spectrogram.shape[1]):
            syl_slice = self.prepare_spectrogram(syl_spectrogram[ ,j])
            label_scores = self.forward(syl_slice)
        return(label_scores)

    # Calculate most likely label given calculate score
    def score_to_label(self, syllable_score):
        syl_index = numpy.argmax(syllable_score.data)
        return(self.syl_to_idx[syl_index])

    # Predict class of syllable spectrogram
    def predict(self, syllable):
        # Need to prepare spectogram
        syllable_score = self.calculate_syllable_score(syllable)
        return(self.score_to_label(syllable_score))

    def calculate_loss(self, syllable_scores, target):
        return(self.loss_function(syllable_scores, target))

    def create_label_dictionary(self):
        syllable_index = {}
        for i, each in enumerate(self.labels):
            syllable_index[i] = each
            syllable_index[each] = i
        return(syllable_index)

    def prepare_label(self, syl_label):
        id_tensor = torch.zeros(1, self.output_dim)
        id_tensor[0][self.labels_to_index[syl_label]] = 1
        return(autograd.Variable(torch.LongTensor(id_tensor)))

    def prepare_spectrogram(self, syl_spectrogram):
        # do from numpy type thing
        return(autograd.Variable(torch.Tensor(syl_spectrogram)))

    def save_model(self, save_file):
        with open(save_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self, read_file):
        with open(read_file, 'rb') as pickle_input:
            self = pickle.load(pickle_input)


    # mabye put this outside of class
    def train_model(self, syllable_collection, n_epochs, save_file=None):
        if not isinstance(syllable_collection, SyllableCollection):
            raise ValueError('Unsupported type <{0}> for syllable_collection. Expected SyllableCollection.'.format(type(syllable_collection)))
        for epoch in range(n_epochs):
            print('{0}/{1} epochs...'.format(epoch, n_epochs))
            labels = syllable_collection.get_labels()
            s_number = 0
            for i in range(syllable_collection.n_syllables):
                current_syl = syllable_collection.syllables[i]
                spectrogram = current_syl.spectrogram
                # Clear gradients and hidden state
                self.zero_grad()
                # Remember, hidden is tuple with both hidden state and
                # cell state as Torch.Variables
                self.hidden = self.init_hidden()

                # Iterate over syllable slices
                # instantiate spectrogram and
                # target label as Torch Variables
                target = self.prepare_label(labels[i])
                label_scores = self.calculate_syllable_score(spectrogram)
                if s_number % 500 == 0:
                    pred_label = self.score_to_label(label_scores)
                    print('Expected: {0} | Predicted: {1}'.format(
                          labels[i], pred_label))

                # compute loss, gradients, and update paramters
                loss = self.calculate_loss(label_scores, target)
                loss.backward()
                self.optimizer.step()
        # Save current model after each epoch
        if save_file is not None:
            print('Saving model after epoch {0}.'.format(epoch))

    # change back to SoundDataSet
    def create_training_and_test(self, fold=10):
        test_size = int(len(self.sound_data.syllables)/fold)
        all_indices = range(self.sound_data.n_syllables)
        test_indices = random.choice(all_indices,
                                     size=test_size, replace=False)
        training_indices = list(set(all_indices).difference(test_indices))
        test_samples = self.sound_data.syllables[list(test_indices)]
        training_samples = self.sound_data.syllables[list(training_indices)]
        return(training_samples, test_samples)
