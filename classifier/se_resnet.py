import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1)
        out = out * original_out
        out += residual
        out = self.relu(out)

        return out


class SE_ResNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], num_classes=26, channel_num=8):
        super(SE_ResNet, self).__init__()
        block = BasicBlock
        self.inplanes = 64
        self.external = 15
        self.conv1 = nn.Conv1d(channel_num, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512 * block.expansion + self.external, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class SE_ResNet_Peak_Detection_Tmp(nn.Module):
    def __init__(self, layers=[3,4,6,3], num_classes=26, channel_num=8, drop_block_size=None, input_length=4992, hidden=512, use_dense=True):
        super(SE_ResNet_Peak_Detection_Tmp, self).__init__()
        block = BasicBlock
        self.kernel_size = 7
        self.input_length = input_length
        self.inplanes = 64
        self.external = 15
        self.conv1 = nn.Conv1d(channel_num, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_block_size=drop_block_size)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_block_size=drop_block_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_block_size=drop_block_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_block_size=drop_block_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if use_dense == False:
            self.mlp = nn.Sequential(
                nn.Linear(8 * 312, hidden),
                nn.ReLU(True),
                nn.Dropout(.2),
                nn.Linear(hidden, 156 * 5 * 3),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Dropout(.2),
                nn.Linear(8 * 312, 156 * 5 * 3),
            )
        # self.mlp = nn.Sequential(
        #     nn.Linear(512 * 156, 156 * 1 * 3)
        # )
        # self.fc = nn.Linear(512 * block.expansion + self.external, num_classes)
        self.branch_conv = nn.Conv1d(256, 8, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1, drop_block_size=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        f = self.layer3(x)

        branch_x = self.branch_conv(f)
        branch_x = branch_x.view(branch_x.size(0), -1)
        y_peak = self.mlp(branch_x)
        y_peak = torch.sigmoid(y_peak)
        y_peak = y_peak.view(-1, 156, 5, 3)

        x = self.layer4(f)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x3 = torch.cat([x, x2], dim=1)
        # x = self.relu(x)
        y_cla = self.fc(x)
        return y_cla, y_peak

def se_resnet(layers=[3, 4, 6, 3], num_classes=26, channel_num=8):
    model = SE_ResNet(layers=layers, num_classes=num_classes, channel_num=channel_num)
    return model

def se_resnet_peak_detection_tmp(layers=[3,4,6,3], num_classes=26, channel_num=8, drop_block_size=None, input_length=4992, hidden=512, use_dense=True):
    model = SE_ResNet_Peak_Detection_Tmp(layers=layers, num_classes=num_classes, channel_num=channel_num, drop_block_size=drop_block_size, input_length=input_length, hidden=hidden, use_dense=use_dense)
    return model

if __name__ == '__main__':
    model = se_resnet(layers=[3, 4, 6, 3], num_classes=26)
    # x = torch.randn((16, 12, 3000))
    x = torch.randn((16, 8, 7500))
    y = model(x)
    print(y.shape)
