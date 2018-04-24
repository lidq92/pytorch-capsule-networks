import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy import prod
import model.modules.capsules as caps

class CapsuleNetwork(nn.Module):
    """
    NIPS 2017
    """
    def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, num_routing, kernel_size=9, out1_features=512, out2_features=1024, use_cuda=True):

        super(CapsuleNetwork, self).__init__()
        self.use_cuda = use_cuda
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size)
        self.relu = nn.ReLU(inplace=True)

        self.primary = caps.PrimaryCapsules(channels, channels, primary_dim, kernel_size)
		
        primary_caps = int(channels / primary_dim * ( img_shape[1] - 2*(kernel_size-1) ) * ( img_shape[2] - 2*(kernel_size-1) ) / 4)
        self.digits = caps.RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing, use_cuda=use_cuda)

        self.decoder = nn.Sequential(
			nn.Linear(out_dim * num_classes, out1_features),
			nn.ReLU(inplace=True),
			nn.Linear(out1_features, out2_features),
			nn.ReLU(inplace=True),
			nn.Linear(out2_features, int(prod(img_shape)) ),
			nn.Sigmoid()
		)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)

        # Reconstruct the *predicted* image
        _, max_length_idx = preds.max(dim=1)
        y = Variable(torch.sparse.torch.eye(self.num_classes))
        if self.use_cuda:
            y = y.cuda()

        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder( (out*y).view(out.size(0), -1) )
        reconstructions = reconstructions.view(-1, *self.img_shape)

        return preds, reconstructions


class CapsNet(nn.Module):
    def __init__(self, in_channels=1, A=32, B=32, C=32, D=32, E=10, r=3, routing='EM_routing', use_cuda=True):
        super(CapsNet, self).__init__()
        """
        ICLR 2018
        """
        self.num_classes = E
        self.use_cuda    = use_cuda
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=A,
                               kernel_size=5, stride=2)
        self.relu  = nn.ReLU()
        self.primary_caps = caps.PrimaryCaps(A, B)
        self.convcaps1 = caps.ConvCaps(B, C, K=3, stride=2, iteration=r,
                                  coordinate_add=False, transform_share=False,
                                       routing=routing, use_cuda=use_cuda)
        self.convcaps2 = caps.ConvCaps(C, D, K=3, stride=1, iteration=r,
                                  coordinate_add=False, transform_share=False,
                                       routing=routing, use_cuda=use_cuda)
        self.classcaps = caps.ConvCaps(D, E, K=0, stride=1, iteration=r,
                                  coordinate_add=True, transform_share=True,
                                       routing=routing, use_cuda=use_cuda)
        self.decoder = nn.Sequential(
            nn.Linear(16 * E, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, lambda_):  # b,1,28,28
        x = self.relu(self.conv(x))  # b,32,12,12
        x = self.primary_caps(x)  # b,32*(4*4+1),12,12
        x = self.convcaps1(x, lambda_)  # b,32*(4*4+1),5,5
        x = self.convcaps2(x, lambda_)  # b,32*(4*4+1),3,3
        p, a = self.classcaps(x, lambda_)  # b,10*16+10
        p = p.squeeze()
        # convert to one hot
        _, max_index = a.max(dim=1)
        y = Variable(torch.sparse.torch.eye(self.num_classes))
        if self.use_cuda:
            y = y.cuda()
        y = y.index_select(dim=0, index=max_index.squeeze()).unsqueeze(2)

        reconstructions = self.decoder((p * y).view(p.size(0), -1))

        return a.squeeze(), reconstructions
