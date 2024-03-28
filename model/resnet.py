import torchvision
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class ResNet50_fc(nn.Module):
    def __init__(self, pretrain=True):
        super(ResNet50_fc, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(pretrained=True)
        else:
            net = models.resnet50()
        self.net = net
        self.net.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.net.fc(x)
        return x


class ResNet50_fc2(nn.Module):
    def __init__(self, pretrain=True, num_class=2):
        super(ResNet50_fc2, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        self.feature_extractor = net
        self.fc = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, num_class, kernel_size=1))

    def forward(self, x, return_feature=False):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)
        feature = x
        feature = F.avg_pool2d(feature, kernel_size=(feature.size(2), feature.size(3)), padding=0)
        feature = feature.view(feature.size(0), -1)
        x = self.fc(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        if return_feature:
            return x, feature
        else:
            return x


class ResNet50_fc_two_heads(nn.Module):
    def __init__(self, pretrain=True, num_class=2, SimCLR=False):
        super(ResNet50_fc_two_heads, self).__init__()
        # resnet50
        if pretrain:
            net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50()
        if SimCLR:
            checkpoint = '/home/ubuntu/data/lanfz/codes/SimCLR/runs/Dec01_10-10-06_c1501-x/checkpoint_best_0090.pth.tar'
            ckpt = torch.load(checkpoint)
            for key in list(ckpt['state_dict'].keys()):
                new_key = key[9:]
                ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(key)
                net.load_state_dict(ckpt['state_dict'], strict=False)
        self.feature_extractor = net
        self.fc1 = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, num_class, kernel_size=1))
        self.fc2 = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, num_class, kernel_size=1))
    def forward(self, x, return_feature=False):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)
        feature = x
        x1 = self.fc1(x)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(x.size(0), -1)
        x2 = self.fc2(x)
        x2= F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2 = x2.view(x.size(0), -1)
        if return_feature:
            return x1, x2, feature
        else:
            return (x1+x2)/2


class ResNet50_simclr(nn.Module):
    def __init__(self, num_class=2):
        super(ResNet50_simclr, self).__init__()
        # resnet50
        net = models.resnet50()
        checkpoint = '/home/ubuntu/data/lanfz/codes/SimCLR/runs/Dec01_10-10-06_c1501-x/checkpoint_latest_0100.pth.tar'
        ckpt = torch.load(checkpoint)
        print(ckpt.keys())
        # print(ckpt['state_dict'].keys())
        for key in list(ckpt['state_dict'].keys()):
            new_key = key[9:]
            ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(key)
        # print(ckpt['state_dict'].keys())
        net.load_state_dict(ckpt['state_dict'], strict=False)
        self.net = net
        self.fc = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(2048, num_class, kernel_size=1))

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        return x

if __name__ == "__main__":
    model = ResNet50_simclr(num_class=5)
    input_tensor = torch.rand((2,3,224,224))
    out = model(input_tensor)
    print(out.shape)
