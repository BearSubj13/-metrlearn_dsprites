import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size):
        #nn.Module.__init__(self)
        super().__init__()

        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.latent_size = latent_size
        self.input_image_size = 64

        ###############
        # ENCODER
        ##############
        self.e_conv_1 = nn.Conv2d(in_channels, dec_channels,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(dec_channels)

        self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels * 2,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(dec_channels * 2)

        self.e_conv_3 = nn.Conv2d(dec_channels * 2, dec_channels * 4,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(dec_channels * 4)

        self.e_conv_4 = nn.Conv2d(dec_channels * 4, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_4 = nn.BatchNorm2d(dec_channels * 8)

        self.e_conv_5 = nn.Conv2d(dec_channels * 8, dec_channels * 16,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_5 = nn.BatchNorm2d(dec_channels * 16)

        self.e_fc_0 = nn.Linear(dec_channels * 16 * 2 * 2, dec_channels * 16 * 2 * 2)
        self.e_bn_6 = nn.BatchNorm1d(dec_channels * 16 * 2 * 2)

        self.e_fc_1 = nn.Linear(dec_channels * 16 * 2 * 2, latent_size)

        ###############
        # DECODER
        ##############

        self.d_fc_1 = nn.Linear(latent_size, dec_channels * 16 * 2 * 2)
        self.d_bn_0 = nn.BatchNorm1d(dec_channels * 16 * 2 * 2)
        self.d_fc_2 = nn.Linear(dec_channels * 16 * 2 * 2, dec_channels * 16 * 2 * 2)

        self.d_conv_1 = nn.Conv2d(dec_channels * 16, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels * 8)

        self.d_conv_2 = nn.Conv2d(dec_channels * 8, dec_channels * 4,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels * 4)

        self.d_conv_3 = nn.Conv2d(dec_channels * 4, dec_channels * 2,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels * 2)

        self.d_conv_4 = nn.Conv2d(dec_channels * 2, dec_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)

        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)

        # Reinitialize weights using He initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def encode(self, x):

        # h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)

        # h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_2(x)

        # h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_3(x)

        # h4
        x = self.e_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_4(x)

        # h5
        x = self.e_conv_5(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_5(x)

        # fc
        x = x.view(-1, self.dec_channels * 16 * 2 * 2)
        x = self.e_fc_0(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_6(x)

        x = self.e_fc_1(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def decode(self, x):

        # h1
        # x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_0(x)

        x = self.d_fc_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = x.view(-1, self.dec_channels * 16, 2, 2)

        # h2
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)

        # h3
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)

        # h4
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)

        # h5
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_4(x)

        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_5(x)
        x = torch.sigmoid(x)

        return x

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if 'e_' in name:
                param.requires_grad = False

    def unfreeze_encoder(self):
        for name, param in self.named_parameters():
            if 'e_' in name:
                param.requires_grad = True

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return z, decoded


class SimpleNet(nn.Module):
    '''
    predicts whether a figure is a square, ellips or heart
    from a latent vector
    '''
    def __init__(self, latent_size, number_of_classes=3):
        #nn.Module.__init__(self)
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 64)
        self.bn_1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, latent_size)
        self.bn_2 = nn.BatchNorm1d(latent_size)
        self.fc3 = nn.Linear(latent_size, int(latent_size/2))
        self.bn_3 = nn.BatchNorm1d(int(latent_size/2))
        self.fc4 = nn.Linear(int(latent_size/2), number_of_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def classification(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.bn_1(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.bn_2(x)

        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.bn_3(x)

        x = self.fc4(x)
        return x

    def forward(self, x):
        x = self.classification(x)
        return x


class ComplexNet(AutoEncoder, SimpleNet):
    def __init__(self, in_channels, dec_channels, latent_size, number_of_classes=3):
        SimpleNet.__init__(self, latent_size=latent_size, number_of_classes=number_of_classes)
        AutoEncoder.__init__(self, in_channels=in_channels, dec_channels=dec_channels, latent_size=latent_size)

    def forward(self, x):
        z = self.encode(x)
        y = self.classification(z)
        return y

