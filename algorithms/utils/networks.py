from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, state_size, action_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        return self.fc3(x.view(x.size(0), -1))


class NatureCNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(NatureCNN, self).__init__()

        kernel_size = 5
        stride = 2
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size, stride), kernel_size, stride),
                                kernel_size, stride)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size, stride), kernel_size, stride),
                                kernel_size, stride)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.fc1(x.view(x.size(0), -1))
