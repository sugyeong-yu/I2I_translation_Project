import torch , torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Residual_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Residual_block,self).__init__()

        self.main=nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True))

    def forward(self,x):
        return x+self.main(x)



class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator,self).__init__()

        self.conv1 = nn.Conv2d(3+c_dim,conv_dim,kernel_size=7,stride=1,padding=3)
        self.InsNorm1 = nn.InstanceNorm2d(conv_dim,affine=True) # affine True : 매개변수도 학습가능하게 해줌
        self.activation=nn.ReLU(inplace=True)# inplace : input으로 들어온것 자체를 수정하겠다는 뜻?

        self.conv2 = nn.Conv2d(conv_dim,conv_dim*2,kernel_size=4,stride=2,padding=1)
        self.InsNorm2 = nn.InstanceNorm2d(conv_dim*2,affine=True)

        self.conv3 = nn.Conv2d(conv_dim*2,conv_dim*4,kernel_size=4,stride=2,padding=1)
        self.InsNorm3 = nn.InstanceNorm2d(conv_dim*4,affine=True)

        self.Downsampling = nn.Sequential(self.conv1,self.InsNorm1,self.activation,
                                          self.conv2,self.InsNorm2,self.activation,
                                          self.conv3,self.InsNorm3,self.activation)

        bottleneck=[]
        for i in range(repeat_num):
            bottleneck.append( Residual_block(conv_dim,conv_dim,repeat_num))
        self.bottlenecks =nn.Sequential(bottleneck)


        self.deconv1 = nn.ConvTranspose2d(conv_dim*4,conv_dim*2,kernel_size=4,stride=2,padding=1)
        self.InsNorm4 = nn.InstanceNorm2d(conv_dim*2,affine=True)

        self.deconv2 = nn.ConvTranspose2d(conv_dim*2,conv_dim,kernel_size=4,stride=2,padding=1)
        self.InsNorm5 = nn.InstanceNorm2d(conv_dim,affine=True)

        self.conv4 = nn.Conv2d(conv_dim,3+c_dim,kernel_size=7,stride=1,padding=3)
        self.tanh = nn.Tanh()#-1~1 사이값으로

        self.Upsampling=nn.Sequential(self.deconv1,self.InsNorm4,self.activation,
                                      self.deconv2,self.InsNorm5,self.activation,
                                      self.conv4,self.tanh)

    def forward(self,x,c):
        # 데이터구조를 알아야할듯. 왜 c와 x 처럼 되는지?
        # c가 사용할 특징?
        c = c.view(c.size(0),c.size(1),1,1)
        c = c.repeat(1,1,x.size(2),x.size(3))
        x = torch.cat([x,c],dim=1)

        G = nn.Sequential(self.Downsampling,self.bottlenecks,self.Upsampling)
        return G(x)

class Discriminator(nn.Module):
    def __init__(self,img_size,conv_dim=64,c_dim=5,repeat_num=6):
        super(Discriminator,self).__init__()

        layers = []
        layers.append(nn.Conv2d(3,conv_dim,kernel_size=4,stride=2,padding=1))
        layers.append(nn.LeakyReLU(inplace=True))
        #hidden layer
        d=conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(d,d*2,kernel_size=4,stride=2,padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            d*=2

        self.main=nn.Sequential(*layers)

        #output layer
        self.src_d=nn.Conv2d(d,1,kernel_size=3,stride=1,padding=1)
        self.cls_d=nn.Conv2d(d,c_dim,kernel_size=int(img_size/64))

    def forward(self,x):
        h=self.main(x)
        out_src=self.src_d(h) # real? or fake?
        out_cls=self.cls_d(h) # what domain?
        return out_src,out_cls.view(out_cls.size(0),out_cls.size(1))
