from rnn_draw.configs import *
from rnn_draw.common_import import *

import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def cos_annealing_lr(initial_lr, cur_epoch, epoch_per_cycle):
    return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2


def adjust_learning_rate(optimizer, base_lr, epoch):
    lr = cos_annealing_lr(base_lr, epoch, CYCLE_EPOCHS)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('In total: {} trainable parameters.\n\n'.format(params))


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class SelfAttentionLSTMNoBias(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(512, 512, num_layers=3, dropout=0, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 340)

        self.encoder1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)
        x = self.encoder1(x)
        x = self.encoder2(x)

        x = self.encoder3(x)
        x = x.permute(0, 2, 1)
        out, (hidden, c) = self.lstm(x)
        embedding, attn_weights = self.attention(out)
        outputs = self.fc(embedding)
        return outputs


class SelfAttentionLSTMBias(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(512, 512, num_layers=3, dropout=0, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 340)

        self.encoder1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)
        x = self.encoder1(x)
        x = self.encoder2(x)

        x = self.encoder3(x)
        x = x.permute(0, 2, 1)
        out, (hidden, c) = self.lstm(x)
        embedding, attn_weights = self.attention(out)
        outputs = self.fc(embedding)
        return outputs


class MyDeepDualLSTM(nn.Module):
    def __init__(self, num_class=340):
        super(MyDeepDualLSTM, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(512, 512, num_layers=3, dropout=0, bidirectional=True, batch_first=True)
        self.logit = nn.Linear(1024 * 3, num_class)

        self.pad = False

    def forward(self, x, length):
        batch_size, T, dim = x.shape
        x = x.permute(0, 2, 1)
        x = self.encoder1(x)
        x = self.encoder2(x)

        x = self.encoder3(x)
        x = x.permute(0, 2, 1)

        if self.pad:
            xx = pack_padded_sequence(x, length, batch_first=True)
            yy, (h, c) = self.lstm(xx)
            y, _ = pad_packed_sequence(yy, batch_first=True)
        else:
            y, (h, c) = self.lstm(x)

        z = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return self.logit(z)


if __name__ == '__main__':
    print('models.py')
