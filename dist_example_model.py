import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Net for classification of FashionMINST dataset
    """
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_size = 100
        self.final_convs = 100
        self.conv1 = nn.Conv2d(1, 30, 5, stride=1)
        self.conv2 = nn.Conv2d(30, 60, 5, stride=1)
        self.conv3 = nn.Conv2d(60, self.final_convs, 3, stride=1)
        self.fc1 = nn.Linear(1 * 1 * self.final_convs, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1 * 1 * self.final_convs)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


