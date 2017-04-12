import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

from Syllable import Syllable
from SoundDataSet import SoundDataSet

torch.manual_seed(1)

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SyllableLSTM, self).__init__()
        # input dim should be the number of mels in a syllable slice
        self.input_size = input_size
        # output dim will be the number of possible labels
        self.output_size = output_size
        # hidden size will be output_size // possible input + output later
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTMCell(self.input_size, self.output_size)

    # Initialize hidden state of model, includes cell state and
    # hidden state as tuple
    def init_hidden(self):
        return(autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
               autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

class SyllableLSTM(object):

    def __init__(self, input_size, possible_labels):
        # input dim should be the number of mels in a syllable slice
        self.input_size = input_size
        # output dim will be the number of possible labels
        self.output_size = len(possible_labels)
        self.labels = possible_labels
        self.label_index_dict = self.create_label_dictionary()
        # empty syllable to place into dictionary
        empty_syl = Syllable(start=0, end=0, species='')
        # dictionary containg 'best' example of each syllable
        self.best_syllables = {each: (empty_syl, 0) for each in self.labels}
        # hidden size will be output_size // possible input + output later
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.lstm = LSTM(input_size=self.input_size,
                         hidden_size=self.output_size,
                         output_size=self.output_size)

    # Calculate most likely label given calculate score
    def score_to_label(self, syllable_score):
        syl_index = syllable_score.data.topk(1)
        return(self.label_index_dict[syl_index])

    # Predict class of syllable spectrogram
    def predict(self, syllable):
        # Need to prepare spectogram
        syllable_score = self.lstm(syllable)
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
        # create 'one-hot' tensor to denote syllable label
        id_tensor = torch.zeros(1, 1, self.output_size)
        id_tensor[0, 0, self.labels_to_index[syl_label]] = 1
        return(autograd.Variable(torch.LongTensor(id_tensor)))

    def prepare_spectrogram(self, spectrogram):
        # convert 126 x seq length spectrogram matrix to tensor
        spec_tensor = torch.from_numpy(spectrogram)
        # transpose because PyTorch likes input in last index and instances
        # of input as first index
        spec_tensor = torch.t(spec_tensor)
        # add filler dimension for mini-batch
        spectorch.unsqueeze(spec_tesnor, 1)
        return(autograd.Variable(spec_tensor))

    def save_model(self, save_file):
        torch.save(self.lstm.state_dict(), save_file)

    def load_model(self, read_file):
        self.lstm.load_state_dict(torch.load(read_file))

    # mabye put this outside of class
    def train_model(self, syllable_collection, n_epochs, save_file=None):
        if not isinstance(syllable_collection, SyllableCollection):
            raise ValueError('Unsupported type <{0}> for syllable_collection. Expected SyllableCollection.'.format(type(syllable_collection)))
        for epoch in range(n_epochs):
            print('{0}/{1} epochs...'.format(epoch, n_epochs))
            labels = syllable_collection.get_labels()
            s_number = 0
            for i in range(syllable_collection.n_syllables):
                target = self.prepare_label(labels[i])
                syllable_input = self.prepare_spectrogram(current_syl_spec)
                current_syl_spec = syllable_collection.syllables[i].spectrogram
                # Clear gradients and hidden state
                self.zero_grad()
                # Remember, hidden is tuple with both hidden state and
                # cell state as Torch.Variables
                self.hidden = self.init_hidden()
                # go through forward step, receive scores
                label_scores = self.lstm(syllable_input)
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

    def test_model(self, syllable_collection):
        if not isinstance(syllable_collection, SyllableCollection):
            raise ValueError('Unsupported type <{0}> for syllable_collection. Expected SyllableCollection.'.format(type(syllable_collection)))

        confusion = numpy.zeros((self.output_size, self.output_size))
        for syl in syllable_collection:
            scores = self.lstm(syl.spectrogram)
            pred_label = self.score_to_label(scores)
            true_index = self.label_index_dict[syl.label]
            pred_index = self.label_index_dict[pred_label]
            confusion[true_index, pred_index] += 1

            match = pred_label == syl.label
            if match and scores[syl_index] > self.best_syllables[syl.label][1]:
                self.best_syllables[syl.label] = (syl, scores[syl_index])
