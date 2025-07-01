import torch
import numpy as np


class CNN(torch.nn.Module):
    def __init_weights(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(w.weight)
            if w.bias is not None:
                torch.nn.init.constant_(w.bias, val=0.0)
    
    def __init__(self, n_classes=4, n_modalities=3, lrelu_slope=.2):
        super(CNN, self).__init__()
        self.n_classes = n_classes
        self.lrelu_slope = lrelu_slope
        self.n_modalities = n_modalities
        
        self.cnn_head = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.n_modalities, out_channels=16, kernel_size=7),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.MaxPool1d(5),

            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.MaxPool1d(3),

            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.AdaptiveMaxPool1d(1),

            torch.nn.Flatten()
        )
        
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.Linear(256, 64),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(self.lrelu_slope),
            torch.nn.Linear(32, self.n_classes)
        )
        
        self.cnn_head.apply(CNN.__init_weights)
        self.classification_head.apply(CNN.__init_weights)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {n_params:,}")


    def forward(self, x):
        z = self.cnn_head(x)
        return self.classification_head(z)


if __name__ == '__main__':
    cnn = CNN(n_classes=3, n_modalities=3)
    print(cnn)
    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"Model parameters: {total_params:,}")
    print(cnn(torch.randn(64, 3, 1000)).shape)
