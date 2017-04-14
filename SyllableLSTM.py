import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

from Syllable import Syllable
from SoundDataSet import SoundDataSet
from Recording import Recording
from SyllableCollection import SyllableCollection

import time

from pandas import DataFrame

class LSTM(nn.Module):

    def __init__(self, input_size, output_size):

        super(LSTM, self).__init__()
        self.hidden_size = output_size
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTMCell(input_size, output_size)
        self.softmax = nn.LogSoftmax()


    def init_hidden(self):
        zero_tensor = torch.zeros(1, self.hidden_size)
        return(autograd.Variable(zero_tensor), autograd.Variable(zero_tensor))


    def forward(self, net_input):
        for i in range(net_input.size()[0]):
            self.hidden = self.lstm(net_input[i], self.hidden)
        scores = self.softmax(self.hidden[0])
        return(scores)

class SyllableLSTM(object):

    def __init__(self, input_size, possible_labels):
        # input dim should be the number of mels in a syllable slice
        self.input_size = input_size
        # output dim will be the number of possible labels
        self.output_size = len(possible_labels)
        self.labels = possible_labels
        self.label_index_dict = self.create_label_dictionary()
        # empty syllable to place into dictionary
        empty_syl = Syllable(start=0, end=0, species='', sound_file=None)
        # dictionary containg 'best' example of each syllable
        self.best_syllables = {each: (empty_syl, 0) for each in self.labels}
        # hidden size will be output_size // possible input + output later
        self.lstm = LSTM(input_size=self.input_size,
                         output_size=self.output_size)
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.lstm.parameters(), lr=0.1)

    # Calculate most likely label given calculate score
    def score_to_label(self, syllable_score):
        syl_index = numpy.argmax(syllable_score.data.numpy())
        return(self.label_index_dict[syl_index])

    # Predict class of syllable spectrogram
    def predict(self, syllable):
        # Need to prepare spectogram
        syllable_score = self.lstm(syllable)
        return(self.score_to_label(syllable_score))

    def create_label_dictionary(self):
        syllable_index = {}
        for i, each in enumerate(self.labels):
            syllable_index[i] = each
            syllable_index[each] = i
        return(syllable_index)

    def prepare_label(self, syl_label):
        # create 'one-hot' tensor to denote syllable label
        id_array = numpy.zeros((1, self.output_size), dtype=int)
        index = self.label_index_dict[syl_label]
        id_array[0, self.label_index_dict[syl_label]] = 1
        return(autograd.Variable(torch.LongTensor(id_array)))

    @staticmethod
    def prepare_spectrogram(spectrogram):
        # convert 126 x seq length spectrogram matrix to tensor
        spec_tensor = torch.from_numpy(spectrogram).float()
        # transpose because PyTorch likes input in last index and instances
        # of input as first index
        spec_tensor = torch.t(spec_tensor)
        # add filler dimension for mini-batch
        spec_tensor = torch.unsqueeze(spec_tensor, 1)
        return(autograd.Variable(spec_tensor))

    def save_model(self, save_file):
        torch.save(self.lstm.state_dict(), save_file)

    def load_model(self, read_file):
        self.lstm.load_state_dict(torch.load(read_file))

    @staticmethod
    def write_to_loss(iter_number, loss, epoch, loss_file):
        out_list = [str(each) for each in [iter_number, loss.data[0], epoch + 1]]
        if iter_number == 0:
            loss_out = open(loss_file, 'w')
            loss_out.write('Step,Loss,Epoch' + '\n')
            loss_out.write(','.join(out_list) + '\n')
            loss_out.close()
        else:
            loss_out = open(loss_file, 'a')
            loss_out.write(','.join(out_list) + '\n')
            loss_out.close()

    @staticmethod
    def __format_time(time_in_seconds):
        seconds = time_in_seconds
        hours = int(time_in_seconds/60**2)
        minutes = int((time_in_seconds - hours*60**2)/60)
        if (hours*60**2 + minutes*60) > 0:
            seconds = int(time_in_seconds) % (hours*60**2 + minutes*60)
        return('{0}h:{1}m:{2}s'.format(hours, minutes, seconds))


    def train_model(self, syllable_collection, n_epochs, save_file=None,
                    loss_file=None, track_time=False):

        if not isinstance(syllable_collection, SyllableCollection):
            raise ValueError('Unsupported type <{0}> for syllable_collection. Expected SyllableCollection.'.format(type(syllable_collection)))

        iter_number = 0
        for epoch in range(n_epochs):
            if track_time:
                t0 = time.time()

            print('{0}/{1} epochs...'.format(epoch + 1, n_epochs))
            syl_number = 0
            for each in syllable_collection:
                print('Current syllable: ' + each.label)
                # Prepare spectrogram and target tensors
                target = self.prepare_label(each.label)
                syllable_input = self.prepare_spectrogram(each.spectrogram)

                # Clear gradients and hidden state
                self.lstm.zero_grad()
                # Remember, hidden is tuple with both hidden state and
                # cell state as Torch.Variables
                self.lstm.hidden = self.lstm.init_hidden()
                # go through forward step, receive scores
                label_scores = self.lstm(syllable_input)
                if syl_number % 500 == 0:
                    pred_label = self.score_to_label(label_scores)
                    print('Expected: {0} | Predicted: {1}'.format(
                          each.label, pred_label))

                # compute loss, gradients, and update paramters
                loss = self.loss_function(label_scores[0], target[0])

                if loss_file is not None and iter_number % 100 == 0:
                    # write iter number, loss, epoch to csv file.
                    self.write_to_loss(iter_number, loss, epoch, loss_file)

                loss.backward()
                self.optimizer.step()

                syl_number += 1
                iter_number += 1

            if track_time:
                print(time.time() - t0)
                run_time = self.__format_time(time.time() - t0)
                print('Epoch {0} run time: {1}'.format(epoch + 1, run_time))
            # Save current model after each epoch
            save_bool = epoch % 10 == 0 or epoch == n_epochs - 1
            if save_file is not None and save_bool:
                print('Saving model after epoch {0}.'.format(epoch + 1))
                self.save_model(save_file)

    def test_model(self, syllable_collection):
        if not isinstance(syllable_collection, SyllableCollection):
            raise ValueError('Unsupported type <{0}> for syllable_collection. Expected SyllableCollection.'.format(type(syllable_collection)))

        confusion = numpy.zeros((self.output_size, self.output_size),
                                dtype=int)
        for syl in syllable_collection:
            scores = self.lstm(self.prepare_spectrogram(syl.spectrogram))
            pred_label = self.score_to_label(scores)
            true_index = self.label_index_dict[syl.label]
            pred_index = self.label_index_dict[pred_label]
            confusion[true_index, pred_index] += 1

            match = pred_label == syl.label
            if match and scores.data[0][true_index] > self.best_syllables[syl.label][1]:
                self.best_syllables[syl.label] = (syl, scores[true_index])
        labels = [self.label_index_dict[i] for i in range(confusion.shape[0])]
        return(DataFrame(confusion, columns=labels, index=labels))

if __name__ == '__main__':
    syl_1 = Syllable(start=12.19, end=12.43, sound_file='Downloads/CATH1.wav')
    syl_2 = Syllable(start=12.60, end=12.85, sound_file='Downloads/CATH1.wav')
    syl_3 = Syllable(start=13.03, end=13.19, sound_file='Downloads/CATH1.wav')

    def prepare_spectrogram(spectrogram):
        # convert 126 x seq length spectrogram matrix to tensor
        spec_tensor = torch.from_numpy(spectrogram).float()
        # transpose because PyTorch likes input in last index and instances
        # of input as first index
        spec_tensor = torch.t(spec_tensor)
        # add filler dimension for mini-batch
        spec_tensor = torch.unsqueeze(spec_tensor, 1)
        return(autograd.Variable(spec_tensor))

    syl1_input = prepare_spectrogram(syl_1.spectrogram)
    syl2_input = prepare_spectrogram(syl_2.spectrogram)
    syl3_input = prepare_spectrogram(syl_3.spectrogram)

    new_model = LSTM(128, 2)
    output = []
    for each in [syl1_input, syl2_input, syl3_input]:
        new_model.init_hidden()
        output.append(new_model(each))

    r1 = Recording('Downloads/CATH2.wav', 'Downloads/CATH2.TextGrid',
                   species='CATH')
    r1.get_annotations()
    syllable_lstm = SyllableLSTM(128, r1.unique_syllable_labels())
    syllable_lstm.train_model(r1.syllables,
                              n_epochs=1,
                              save_file='test_model.pkl',
                              loss_file='test_loss.csv',
                              track_time=True)
    c_matrix = syllable_lstm.test_model(r1.syllables)
    c_matrix.to_csv('cmatrix.csv')
