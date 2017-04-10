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

    def __init__(self, sound_data_set, n_epochs):
        super(SyllableLSTM, self).__init__()
        self.sound_data = sound_data_set
        self.input_dim = self.sound_data.syllables[0].spectrogram.shape[0]
        self.output = self.sound_data.syllables.get_unique_syllables()
        self.output_dim = len(self.output)
        self.syl_to_idx = self.syllable_to_index()
        # hidden dim wiill be output_dim
        self.n_epochs = n_epochs
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.input_dim, self.output_dim)
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

    def init_hidden(self):
        return(autograd.Variable(torch.zeros(1, 1, self.output_dim)))

    def forward(self, syllable_slice):
        lstm_out, self.hidden = self.lstm(syllable_slice, 1, -1)
        syllable_scores = F.log_softmax(lstm_out)
        return(syllable_scores)

    def calculate_syllable_score(self, syllable):
        syl_spectrogram = syllable.spectrogram
        for j in range(syl_spectrogram.shape[1]):
            syl_slice = self.prepare_spectrogram(syl_spectrogram[ ,j])
            label_scores = self(syl_slice)
        return(label_scores)

    def score_to_label(self, syllable_score):
        syl_index = numpy.argmax(syllable_score.data)
        return(self.syl_to_idx[syl_index])

    def predict(self, syllable):
        syllable_score = self.calculate_syllable_score(syllable)
        return(self.score_to_label(syllable_score))

    def train_model(self):
        for epoch in range(self.n_epochs):
            print('{0}/{1} epochs...'.format(epoch, self.n_epochs))
            labels = self.sound_data.syllables.get_labels()
            s_number = 0
            for i in range(self.sound_data.n_syllables):
                syl = self.sound_data.syllables[i]

                # Clear gradients and hidden state
                self.zero_grad()
                self.hidden = self.init_hidden()

                # Iterate over syllable slices, instantiate spectrogram and
                # target label as Torch Variables
                target = self.prepare_label(labels[i])
                label_scores = self.calculate_syllable_score(syl)
                if s_number % 500 == 0:
                    pred_label = self.score_to_label(label_scores)
                    print('Expected: {0} | Predicted: {1}'.format(
                          labels[i], pred_label))

                # compute loss, gradients, and update paramters
                loss = self.calculate_loss(label_scores, target)
                loss.backward()
                self.optimizer.step()

    def calculate_loss(self, syllable_scores, target):
        return(self.loss_function(syllable_scores, target))

    def create_training_and_test(self, fold=10):
        test_size = int(len(self.sound_data.syllables)/fold)
        all_indices = range(self.sound_data.n_syllables)
        test_indices = random.choice(all_indices, size=test_size, replace=False)
        training_indices = list(set(all_indices).difference(test_indices))
        test_samples = self.sound_data.syllables[list(test_indices)]
        training_samples = self.sound_data.syllables[list(training_indices)]
        return(training_samples, test_samples)

    def syllable_to_index(self):
        syllable_index = {}
        for i, each in enumerate(self.output):
            syllable_index[i] = each
            syllable_index[each] = i
        return(syllable_index)

    def prepare_label(self, syl_label):
        id_vec = numpy.zeros(self.output_dim)
        id_vec[self.syl_to_idx[syl_label]] = 1
        tensor = torch.LongTensor(id_vec)
        return(autograd.Variable(tensor))

    def prepare_spectrogram(self, syl_spectrogram):
        return(autograd.Variable(torch.Tensor(syl_spectrogram)))

    def predict(self, syllable):
        if not isinstance(syllable, Syllable):
            raise ValueError('Cannot predict on {0} type. Must be Syllable.'.format(syllable))

    def save_model(self, save_file):
        with open(save_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self, read_file):
        with open(read_file, 'rb') as pickle_input:
            self = pickle.load(pickle_input)
