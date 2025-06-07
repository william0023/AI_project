import torch
from torch import nn
from torchstat import stat
import torchsummary
# from nni.compression.pytorch.utils.counter import count_flops_params

# Total params: 8,571,666
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Total memory: 702.50MB
# Total MAdd: 67.3GMAdd
# Total Flops: 33.05GFlops
# Total MemR+W: 1.42GB


class unet_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,  padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channel=3, num_classes=2, ):
        super(Unet, self).__init__()
        self.conv1 = unet_block(in_channel, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = unet_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = unet_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = unet_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = unet_block(256, 512)
        self.pool5 = nn.MaxPool2d(2)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = unet_block(512+256, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = unet_block(256+128, 128)
        self.up8 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.conv8 = unet_block(128+64, 64)
        self.up9 = nn.ConvTranspose2d(64, 32,   kernel_size=2, stride=2)
        self.conv9 = unet_block(64+32, 32)
        self.up10 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv10 = unet_block(32+16, 32)
        self.conv_out = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        c3 = self.conv3(x)
        x = self.pool3(c3)
        c4 = self.conv4(x) # 256. 64. 64
        x = self.pool4(c4) # 256.32.32
        c5 = self.conv5(x) # 512.32.32
        x = self.pool5(c5)
        # print(x.shape)

        x = self.up6(x) # 256. 32. 32
        x = torch.cat([x, c5], dim=1)
        x = self.conv6(x)
        x = self.up7(x)  #128.64.64
        # print(x.shape)

        x = torch.cat([x, c4], dim=1)
        x = self.conv7(x)
        x = self.up8(x)
        x = torch.cat([x, c3], dim=1)
        x = self.conv8(x)
        x = self.up9(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv9(x)
        x = self.up10(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv10(x)
        x = self.conv_out(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    model = Unet()
    y = model(x)
    print(y)
    print(y.shape)
    # stat(model, (3, 512, 512))
    # y=x.reshape(x.shape[0],-1)




##############  参数计算 ##################
    # model = Unet()
    # dummy_input = torch.randn(1, 3, 512, 512)
    # flops, params, results = count_flops_params(model, dummy_input)
    # torchsummary.summary(model.cpu(), (3, 512, 512))
    # print('parameters_count:', count_parameters(model))

    # print(model)
    # y = model(x)
    #print(y)
    # print(y.shape)
    # stat(model, (3, 512, 512))

# if __name__ == '__main__':
#     x = torch.randn(1,3, 572, 572)
#    # print(x)
#     model = Unet()
#     y = model(x)
#     print(y)

