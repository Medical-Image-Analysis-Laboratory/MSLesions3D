'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
warnings.filterwarnings(action="ignore", message=".*TracerWarning.*")


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if out.isnan().sum() > 0:
            breakpoint()
            raise Exception("NaN Loss in MobileNet Block")
        return out


class MobileNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes=3, width_mult=1.):
        super(MobileNet, self).__init__()

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # channel, n, stride
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(in_channels, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )


    def forward(self, image):
        out = image
        for i, feat in enumerate(self.features):
            print(i, out.shape)
            out = feat(out)
        
        print(i, out.shape)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class LMobileNetBase(pl.LightningModule):
    def __init__(self, in_channels = 3, num_classes=3, width_mult=1.):
        super(LMobileNetBase, self).__init__()

        print(width_mult)

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # channel, n, stride
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(in_channels, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )


    def forward(self, image):
        out = image
        for i, feat in enumerate(self.features):
            # print(i, out.shape)
            out = feat(out)
        
        # print(i, out.shape)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
    def training_step(self, batch):
        img, seg = batch["img"], batch['seg']
        gt = torch.zeros((img.shape[0],1))
        for i in range(seg.shape[0]):
            segun = seg[i].unique()
            prl_presence = torch.FloatTensor([int(len(torch.where(segun < 2000)[0]) > 1)])
            gt[i,0] = prl_presence.item() 
        gt = gt.cuda()
        pred = self(img)
        pred = pred.cuda()
        loss = F.mse_loss(pred, gt)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    in_channels = 1
    num_classes = 1
    model = LMobileNetBase(in_channels=in_channels,num_classes=num_classes, width_mult=1.)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    # print(model)
    
    batch_size = 8
    
    input_var = torch.autograd.Variable(torch.randn(batch_size, in_channels, 250, 300, 300).cuda())
    output = model(input_var)
    loss = (torch.randn((batch_size,num_classes)).cuda() - output).pow(2).sum()
    loss.backward()
    print(output.shape)
