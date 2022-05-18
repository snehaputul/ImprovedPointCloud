import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=1.0, num_classes=10):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6
        self.num_classes = num_classes

    def forward(self, input, target):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid
        probs = torch.sigmoid(input)
        log_probs = -torch.log(probs)
        target = torch.nn.functional.one_hot(target, self.num_classes)

        focal_loss = torch.sum(torch.pow(1 - probs + self.eps, self.gamma).mul(log_probs).mul(target), dim=1)
        # bce_loss = torch.sum(log_probs.mul(target), dim = 1)

        return focal_loss.mean()  # , bce_loss