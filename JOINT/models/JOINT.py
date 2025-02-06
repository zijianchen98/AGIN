import torch
import torch.nn as nn
import torchvision.models as models
from ot.lp import wasserstein_1d
import numpy as np
import torch.nn.functional as F

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]

        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()


def downsample(img1, img2, maxSize = 256):
    _,channels,H,W = img1.shape
    f = int(max(1,np.round(max(H,W)/maxSize)))

    aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
    img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
    img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    # For an extremely Large image, the larger window will use to increase the receptive field.
    if f >= 5:
        win = 16
    else:
        win = 4
    return img1, img2, win, f

def ws_distance(X,Y,P=2,win=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]
    X_sum = X.sum().sum()
    Y_sum = Y.sum().sum()

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    X_1D_pdf = X_1D / (X_sum + 1e-6)
    Y_1D_pdf = Y_1D / (Y_sum + 1e-6)

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = X_1D * X_1D_pdf
    Y_pdf = Y_1D * Y_1D_pdf

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w
    # final = wsd

    return final.sum()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Model_SwinT(torch.nn.Module):
    def __init__(self):
        super(Model_SwinT, self).__init__()
        model = models.swin_t(weights='Swin_T_Weights.DEFAULT')
        model.head = Identity()
        self.feature_extraction = model
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # quality regressor
        self.quality = self.quality_regression(768, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = F.relu(self.custom_layer(x))
        x = self.quality(x)
        return x


# version of VGG16
class DeepWSD(torch.nn.Module):

    def __init__(self, channels=3, load_weights=True):
        assert channels == 3
        super(DeepWSD, self).__init__()
        self.window = 4

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, window, f = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0
        layer_score=[]

        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            tmp = ws_distance(feats0_k, feats1_k, win=window)
            layer_score.append(torch.log(tmp + 1))
            score = score + tmp
        score = score / (k+1)


        if as_loss:
            return score

        elif f==1:
            return torch.log(score + 1)
        else:
            return torch.log(score + 1)**2

class Model_Resnet50(torch.nn.Module):
    def __init__(self):
        super(Model_Resnet50, self).__init__()
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        self.feature_extraction = model
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality = self.quality_regression(1000, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = F.relu(self.custom_layer(x))
        x = self.quality(x)
        return x


class JOINT_Model(torch.nn.Module):
    def __init__(self, pretrained_path_T=None, pretrained_path_R=None):

        super(JOINT_Model, self).__init__()
        # technical feature extractor
        swin_t_technical = Model_SwinT()
        if pretrained_path_T != None:
            print('load aesthetics model')
            swin_t_technical.load_state_dict(torch.load(pretrained_path_T))

        # rationality feature extractor
        resnet_rationality = Model_Resnet50()
        if pretrained_path_R != None:
            print('load distortion model')
            resnet_rationality.load_state_dict(torch.load(pretrained_path_R))


        self.technical_feature_extraction = swin_t_technical.feature_extraction
        self.rationality_feature_extraction = resnet_rationality.feature_extraction
        
        self.quality_T = self.quality_regression(768, 128, 1)
        self.quality_R = self.quality_regression(1000, 128, 1) 
        
    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x_technical, x_rationality):

        x_technical = self.technical_feature_extraction(x_technical)
        x_rationality = self.rationality_feature_extraction(x_rationality)
       

        x_T = self.quality_T(x_technical)
        x_R = self.quality_R(x_rationality)

        return x_T, x_R
