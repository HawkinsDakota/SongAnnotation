class TestLSTM(nn.Module):

    def __init__(self, input_size, output_size):

        super(TestLSTM, self).__init__()

        self.hidden_size = output_size
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(input_size, output_size)

    def init_hidden(self):
        zero_tensor = torch.DoubleTensor(1, 1, self.hidden_size).zero_()
        return(zero_tensor, zero_tensor)


    def forward(self, net_input):
        out, self.hidden = self.lstm(net_input, self.hidden)
        scores = F.log_softmax(out)
        return(scores)
