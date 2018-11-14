from quick_draw.common_import import *
from quick_draw.senet import se_resnext50_32x4d


class SEResNeXt50(nn.Module):
    def __init__(self, dropout=0.2, pretrained='imagenet', input_size=64):
        super().__init__()
        self.model = se_resnext50_32x4d(pretrained=pretrained)
        self.features = self.model.features
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(input_size // 32, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.last_linear = nn.Linear(2048, 340)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)

        return x


class OrdinaryResNet(nn.Module):
    def __init__(self, encoder_depth, dropout_2d=0.2,
                 pretrained=True):
        super().__init__()
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 34, 101, 152 version of ResNet are implemented')

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )  # 64
        self.encoder2 = self.encoder.layer1  # 64
        self.encoder3 = self.encoder.layer2  # 128
        self.encoder4 = self.encoder.layer3  # 256
        self.encoder5 = self.encoder.layer4  # 512
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(4, stride=1, padding=0)

        self.last_linear = nn.Linear(512, 340)

    def forward(self, x):
        x = self.conv1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x


def run_check_net(model, device=DEVICE):
    batch_size = 32
    C, H, W = 3, 64, 64
    num_class = 10

    ipt = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(num_class, batch_size).astype(np.float32)

    ipt = torch.from_numpy(ipt).float().to(device)
    truth = torch.from_numpy(truth).long().to(device)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # see if it can converge ...
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.0001)

    optimizer.zero_grad()
    for i in range(101):
        optimizer.zero_grad()
        logit = model(ipt)
        loss = criterion(logit, truth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        if i % 25 == 0:
            print(f'iter --> {i}, loss --> {loss.item():.6f}')


if __name__ == '__main__':
    print('Checking SEResNeXt50...')
    net = SEResNeXt50(dropout=0.2, pretrained='imagenet', input_size=64).to(DEVICE)
    run_check_net(net)

    print('\nChecking Ordinary ResNet...')
    net = OrdinaryResNet(34).to(DEVICE)
    run_check_net(net)
