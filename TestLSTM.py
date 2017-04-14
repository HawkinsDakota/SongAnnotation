import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

class TestLSTM(nn.Module):

    def __init__(self, input_size, output_size):

        super(TestLSTM, self).__init__()

        self.hidden_size = output_size
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTMCell(input_size, output_size)
        self.softmax = nn.Softmax()

    def init_hidden(self):
        zero_tensor = torch.zeros(1, self.hidden_size)
        return(autograd.Variable(zero_tensor), autograd.Variable(zero_tensor))


    def forward(self, net_input):
        for i in range(net_input.size()[0]):
            self.hidden = self.lstm(net_input[i], self.hidden)
        scores = self.softmax(self.hidden[0])
        return(scores)

def prepare_spectrogram(spectrogram):
    # convert 126 x seq length spectrogram matrix to tensor
    spec_tensor = torch.from_numpy(spectrogram).float()
    # transpose because PyTorch likes input in last index and instances
    # of input as first index
    spec_tensor = torch.t(spec_tensor)
    # add filler dimension for mini-batch
    spec_tensor = torch.unsqueeze(spec_tensor, 1)
    return(autograd.Variable(spec_tensor))

from Syllable import Syllable
syl_1 = Syllable(start=12.19, end=12.43, sound_file='Downloads/CATH1.wav')
syl_2 = Syllable(start=12.60, end=12.85, sound_file='Downloads/CATH1.wav')
syl_3 = Syllable(start=13.03, end=13.19, sound_file='Downloads/CATH1.wav')
print(syl_1, syl_2, syl_3)


syl1_input = prepare_spectrogram(syl_1.spectrogram)
syl2_input = prepare_spectrogram(syl_2.spectrogram)
syl3_input = prepare_spectrogram(syl_3.spectrogram)
hx = autograd.Variable(torch.zeros(1, 2))
cx = autograd.Variable(torch.zeros(1, 2))
output = []

model = nn.LSTMCell(128, 2)
for each in [syl1_input, syl2_input, syl3_input]:
    seq_length = each.size()[0]
    for i in range(seq_length):
        hx, cx = model(each[i], (hx, cx))
    output.append(hx)

new_model = TestLSTM(128, 2)
new_model(syl1_input)
