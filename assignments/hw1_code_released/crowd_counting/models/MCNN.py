import torch
import torch.nn as nn
import math

class subnet(nn.Module):
    def __init__(self, dim_in, dim_out, k1, s1, p1, k2, s2, p2):
        super(subnet, self).__init__()
        self.MaxPool = nn.MaxPool2d(2, 2, 0)
        self.Convlayer1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out*2, k1, s1, p1),
            nn.BatchNorm2d(dim_out*2),
            nn.ReLU(inplace=True)
        )
        self.Convlayer2 = nn.Sequential(
            nn.Conv2d(dim_out*2, dim_out*4, k2, s2, p2),
            nn.BatchNorm2d(dim_out*4),
            nn.ReLU(inplace=True)
        )
        self.Convlayer3 = nn.Sequential(
            nn.Conv2d(dim_out*4, dim_out*2, k2, s2, p2),
            nn.BatchNorm2d(dim_out*2),
            nn.ReLU(inplace=True)
        )
        self.Convlayer4 = nn.Sequential(
            nn.Conv2d(dim_out*2, dim_out, k2, s2, p2),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Convlayer1(x)
        out = self.MaxPool(out)
        out = self.Convlayer2(out)
        out = self.MaxPool(out)
        out = self.Convlayer3(out)
        out = self.Convlayer4(out)
        return out

class MCNN(nn.Module):
    def __init__(self, ):
        super(MCNN, self).__init__()
        self.subnet1 = subnet(3,8,9,1,4,7,1,3)
        self.subnet2 = subnet(3,10,7,1,3,5,1,2)
        self.subnet3 = subnet(3,12,5,1,2,3,1,1)
        self.Convout = nn.Conv2d(30, 1, 1, 1, 0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        featmap1 = self.subnet1(x)
        featmap2 = self.subnet2(x)
        featmap3 = self.subnet3(x)
        out = torch.cat([featmap1,featmap2,featmap3],dim=1)
        out = self.Convout(out)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    device = 'cuda:0'
    input = torch.ones(size=(1, 3, 1024, 681)).to(device)
    model = MCNN().to(device)
    output = model(input)
    print(output.size())
    print([math.ceil(1024/4), math.ceil(681/4)])