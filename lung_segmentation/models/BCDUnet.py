import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,input_size,out_channel):
        super(Conv_Block,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(input_size,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    
    def forward(self,x):
        return self.layer(x)
    
class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)    #1*1的卷积不会进行特征提取，只是为了降通道
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')   #将输入尺寸放大两倍
        out=torch.cat((self.layer(up),feature_map),dim=1)

        return out

# class ConvLSTMCell(nn.Module):
#     def __init__(self,input_size,hidden_size):
#         super(ConvLSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         # Convolutional operations for input-to-state and state-to-state transitions
#         self.conv_i = nn.Conv2d(input_size +hidden_size,hidden_size, 3, 1, 1,padding_mode='reflect',bias=False)
#         self.conv_f = nn.Conv2d(input_size +hidden_size,hidden_size, 3, 1, 1,padding_mode='reflect',bias=False)
#         self.conv_c = nn.Conv2d(input_size +hidden_size,hidden_size, 3, 1, 1,padding_mode='reflect',bias=False)
#         self.conv_o = nn.Conv2d(input_size +hidden_size,hidden_size, 3, 1, 1,padding_mode='reflect',bias=False)

#     def forward(self, x, h, c):
        
#         combined = torch.cat((x, h), dim=1)
#         # Input gate
#         i = torch.sigmoid(self.conv_i(combined))
#         # Forget gate
#         f = torch.sigmoid(self.conv_f(combined))
#         # Memory cell
#         c = f * c + i * torch.tanh(self.conv_c(combined))
#         # Output gate
#         o = torch.sigmoid(self.conv_o(combined))
#         # Hidden state
#         h = o * torch.tanh(c)
#         return h, c
    
class BiConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_channel):
        super(BiConvLSTM, self).__init__()
        self.out_channel = out_channel
        self.conv_lstm = nn.LSTMCell(input_size, hidden_size)
        self.conv = nn.Conv2d(2 , self.out_channel, 1, 1)  # 降通道
        self.relu=nn.LeakyReLU()

    def forward(self, x): 
        
        # print(x.size())
        batch_size, in_channel, height, width = x.size()
        # 初始化前向和后向方向的隐藏状态和记忆单元
        h_forward, c_forward = torch.zeros_like(x[:, 0, :, :]), torch.zeros_like(x[:, 0, :, :])
        h_backward, c_backward = torch.zeros_like(x[:, 0, :, :]), torch.zeros_like(x[:, 0, :, :])
        # print(h_forward.shape)
        # 使用 nn.LSTM 进行前向和后向方向的计算
        for t in range(in_channel):
            # print(x[:, t, :, :].shape)
            # print(x[:, t, :, :].reshape(batch_size,-1).shape)
            h_forward, c_forward = self.conv_lstm(x[:, t, :, :].reshape(batch_size,-1), (h_forward.reshape(batch_size,-1), c_forward.reshape(batch_size,-1)))
        for t in reversed(range(in_channel)):
            h_backward, c_backward = self.conv_lstm(x[:, t, :, :].reshape(batch_size,-1), (h_backward.reshape(batch_size,-1), c_backward.reshape(batch_size,-1)))
            
        y = torch.stack((h_forward, h_backward))
        # print(y.shape)
        y=y.reshape(batch_size,2,height,width)
        # print(y.shape)
        y = self.relu(self.conv(torch.tanh(y)))

        return y

class BCDUnet(nn.Module):
    def __init__(self):
        super(BCDUnet,self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.c5=Conv_Block(512,512)
        self.c6=Conv_Block(512,512)
        self.c7=Conv_Block(512*2,512)
        self.c8=Conv_Block(512*3,512)
        self.u1=UpSample(512)
        self.b1=BiConvLSTM(16*16,16*16,256)
        self.c9=Conv_Block(256,256)
        self.u2=UpSample(256)
        self.b2=BiConvLSTM(1024,1024,128)
        self.c10=Conv_Block(128,128)
        self.u3=UpSample(128)
        self.b3=BiConvLSTM(4096,4096,64)
        self.c11=Conv_Block(64,64)
        self.c12=Conv_Block(64,64)
        self.out=nn.Conv2d(64,3,3,1,1,padding_mode='reflect',bias=False)
        self.TH=nn.Sigmoid()

    def forward(self,x):
        R1=self.c1(x)
        print(R1.shape)
        R2=self.c2(self.d1(R1))
        print(R2.shape)
        R3=self.c3(self.d2(R2))
        print(R3.shape)
        R4=self.c4(self.d3(R3))
        print(R4.shape)
        R5=self.c5(R4)
        print(R5.shape)
        R6=self.c6(R5)
        print(R6.shape)
        R7=self.c7(torch.cat((R5,R6),dim=1))
        print(R7.shape)
        R8=self.c8(torch.cat((R5,R6,R7),dim=1))
        print(R8.shape)
        R9=F.interpolate(self.b1.forward(F.interpolate(self.u1(R8,R3),size=(16,16), mode='bilinear', align_corners=False)),size=(128,128),mode='bilinear', align_corners=False)
        # print(R9.shape)
        R10=F.interpolate(self.b2.forward(F.interpolate(self.u2(R9,R2),size=(32,32), mode='bilinear', align_corners=False)),size=(256,256),mode='bilinear', align_corners=False)
        # print(R10.shape)
        R11=F.interpolate(self.b3.forward(F.interpolate(self.u3(R10,R1),size=(64,64), mode='bilinear', align_corners=False)),size=(512,512),mode='bilinear', align_corners=False)
        return self.TH(self.out(self.c12(self.c11(R11))))


if __name__ == '__main__':
    # x=torch.randn(2,3,512,512)
    # net=BCDUnet()
    # print(net(x).shape)
    x=torch.randn(2,3,512,512)
    net=BCDUnet()
    print(net(x).shape)

