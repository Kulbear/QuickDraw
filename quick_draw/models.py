from quick_draw.configs import *
from quick_draw.common_import import *

from pretrainedmodels import dpn68, dpn92, dpn98, dpn107, dpn131
from pretrainedmodels import xception, polynet, inceptionv4, inceptionresnetv2
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d

import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def cos_annealing_lr(initial_lr, cur_epoch, epoch_per_cycle):
    return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2


def adjust_learning_rate(optimizer, base_lr, epoch):
    lr = cos_annealing_lr(base_lr, epoch, CYCLE_EPOCHS)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class GeneralizedDPN(nn.Module):
    def __init__(self, dpn_version='dpn68', input_size=128, num_classes=340, pretrained='imagenet', dropout_rate=0.):
        super(GeneralizedDPN, self).__init__()
        if dpn_version == 'dpn68':
            self.pretrained_model = dpn68(pretrained=pretrained)
            self.fc_input = 832
        elif dpn_version == 'dpn92':
            self.pretrained_model = dpn92(pretrained='imagenet+5k' if pretrained else None)  # TODO: ugly hard coded
            self.fc_input = 2688
        elif dpn_version == 'dpn98':
            self.pretrained_model = dpn98(pretrained=pretrained)
            self.fc_input = 2688
        elif dpn_version == 'dpn107':
            self.pretrained_model = dpn107(pretrained='imagenet+5k' if pretrained else None)  # TODO: ugly hard coded
            self.fc_input = 2688
        elif dpn_version == 'dpn131':
            self.pretrained_model = dpn131(pretrained=pretrained)
            self.fc_input = 2688
        else:
            raise NotImplementedError('No implementation')
        self.features = self.pretrained_model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 32, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.last_linear = nn.Linear(self.fc_input, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class GeneralizedXception(nn.Module):
    def __init__(self, input_size=128, num_classes=340, pretrained='imagenet', dropout_rate=0.):
        super(GeneralizedXception, self).__init__()
        self.pretrained_model = xception(pretrained=pretrained)
        self.features = self.pretrained_model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 32, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class SEResNeXt(nn.Module):
    def __init__(self, senet_version='se_resnext50_32x4d', input_size=128, num_classes=340, pretrained='imagenet',
                 dropout=0.):
        super().__init__()
        if senet_version == 'se_resnext50_32x4d':
            self.model = se_resnext50_32x4d(pretrained=pretrained)
        elif senet_version == 'se_resnext101_32x4d':
            self.model = se_resnext101_32x4d(pretrained=pretrained)
        self.features = self.model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 32, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)

        return x


class InceptionV4(nn.Module):
    def __init__(self, input_size=128, num_classes=340, pretrained='imagenet',
                 dropout=0.):
        super().__init__()
        self.model = inceptionv4(num_classes=1000, pretrained=pretrained)
        self.features = self.model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 64, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.last_linear = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)

        return x


class InceptionResNetV2(nn.Module):
    def __init__(self, input_size=128, num_classes=340, pretrained='imagenet',
                 dropout=0.):
        super().__init__()
        self.model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
        self.features = self.model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 64, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.last_linear = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)

        return x


class PolyNet(nn.Module):
    def __init__(self, input_size=128, num_classes=340, pretrained='imagenet',
                 dropout=0.):
        super().__init__()
        self.model = polynet(num_classes=1000, pretrained=pretrained)
        self.features = self.model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 64, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)

        return x


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('In total: {} trainable parameters.\n\n'.format(params))


def run_check_net(model, device=DEVICE):
    batch_size = 150
    C, H, W = 3, 128, 128
    num_class = 340

    ipt = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(num_class, batch_size).astype(np.float32)

    ipt = torch.from_numpy(ipt).float().to(device)
    truth = torch.from_numpy(truth).long().to(device)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # see if it can converge ...
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001)
    for i in range(11):
        optimizer.zero_grad()
        logit = model(ipt)
        loss = criterion(logit, truth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if i % 2 == 0:
            print(f'iter --> {i}, loss --> {loss.item():.6f}')


if __name__ == '__main__':
    try:
        print('=================\nChecking InceptionResNetV2...')
        model = InceptionResNetV2(input_size=128, num_classes=340, pretrained='imagenet').to(DEVICE)
        count_trainable_parameters(model)
        run_check_net(model)
    except:
        print('=================\nInceptionResNetV2 failed due to OOM.')

    for dpn in ['dpn68', 'dpn92', 'dpn98', 'dpn107', 'dpn131']:
        try:
            print('=================\nChecking {}...'.format(dpn))
            model = GeneralizedDPN(dpn_version=dpn, num_classes=340, pretrained='imagenet').to(DEVICE)
            count_trainable_parameters(model)
            run_check_net(model)
        except:
            print('{} failed due to OOM.'.format(dpn))

    # try:
    #     print('=================\nChecking Xception...')
    #     model = GeneralizedXception(input_size=128, num_classes=340, pretrained='imagenet').to(DEVICE)
    #     count_trainable_parameters(model)
    #     run_check_net(model)
    # except Exception as e:
    #     print(e)
    #     print('Xception failed due to OOM.')

    for senet in ['se_resnext50_32x4d', 'se_resnext101_32x4d']:
        try:
            print('=================\nChecking {}...'.format(senet))
            model = SEResNeXt(senet_version=senet, num_classes=340, pretrained='imagenet').to(DEVICE)
            count_trainable_parameters(model)
            run_check_net(model)
        except:
            print('{} failed due to OOM.'.format(senet))

    try:
        print('=================\nChecking PolyNet...')
        model = PolyNet(input_size=128, num_classes=340, pretrained='imagenet').to(DEVICE)
        count_trainable_parameters(model)
        run_check_net(model)
    except:
        print('=================\nPolyNet failed due to OOM.')

    try:
        print('=================\nChecking InceptionV4...')
        model = InceptionV4(input_size=128, num_classes=340, pretrained='imagenet').to(DEVICE)
        count_trainable_parameters(model)
        run_check_net(model)
    except:
        print('InceptionV4 failed due to OOM.')
