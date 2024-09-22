import torch
from torch import nn
from torch.nn import functional as F
# from d2l import torch as d2l

class CPA(nn.Module):   #粗粒度补丁注意力
    def __init__(self,channels,H,W,pool_size=6):    #pool_size->计算得到池化窗口大小
        super(CPA,self).__init__()
        self.channels=channels
        self.H=H
        self.W=W
        self.pool_size=pool_size
        self.mlp=nn.Sequential(
            nn.Linear(pool_size*pool_size,18),
            nn.ReLU(),
            nn.Linear(18,pool_size*pool_size),
            nn.Sigmoid()
        )
    def forward(self,x):
        batch_size,channels,H,W=x.size()
        # 对通道进行平均池化
        Fc=torch.mean(x,dim=1,keepdim=True)
        # 生成patch标识符
        Fg=F.avg_pool2d(Fc,kernel_size=(H//self.pool_size,W//self.pool_size))
        #扁平化塞到mlp中  [batch_size,pool_size*pool_size]
        Fg_flat=Fg.view(batch_size,-1)
        #注意力权重
        omega=self.mlp(Fg_flat)
        omega=omega.view(batch_size,1,self.pool_size,self.pool_size)
        #插值上采样(双线性插值，对齐角点)
        omega_up=F.interpolate(omega,size=(H,W),mode='bilinear',align_corners=True)
        #生成最终的注意力图
        A=omega_up*Fc   #元素级乘法
        Y=A+x   #残差连接
        return Y
    
class CFF(nn.Module):   #跨维度特征融合
    def __init__(self,in_channels_2d,in_channels_3d,pool_size=8,deacay_rate=2): 
        #decay_rate控制mlp隐藏层的通道数
        super(CFF,self).__init__()
        #3d卷积将3d特征图降为单通道
        self.conv1=nn.Conv3d(in_channels_3d,1,kernel_size=1)
        #2d卷积将3d特征图重塑为与2d特征图通道一致的3d变换特征图
        self.conv2=nn.Conv2d(16,in_channels_2d,kernel_size=3,padding=1) #16是3D图像的depth
        # mlp用于建模通道之间的关系
        self.fc1=nn.Linear(in_channels_2d*2,in_channels_2d//deacay_rate)
        self.fc2=nn.Linear(in_channels_2d//deacay_rate,in_channels_2d)
    def forward(self,x2d,x3d):
        Batch_size,channels_2d,height,width=x2d.size()  
        Batch_size_3d,channels_3d,height_3d,width_3d,depth_3d=x3d.size()   
        # 3d特征图降维处理
        x3d_c=self.conv1(x3d)   #(B,1,H,W,D)    
        # print(x3d_c.size())
        x3d_s=torch.squeeze(x3d_c,dim=1)    #(B,H,W,D)
        # print(x3d_s.size())
        x3d_r=x3d_s.permute(0,3,1,2)    #(B,D,H,W)
        # print(x3d_r.size())
        x3d_r=x3d_s.reshape(Batch_size,depth_3d,height_3d,width_3d) #(B,D,H,W)   
        # print(x3d_r.size())
        x3d_tfm=self.conv2(x3d_r)   #(B,C_2d,H,W)
        # print(x3d_tfm.size())
        
        #全局池化
        global_avg_2d=F.adaptive_avg_pool2d(x2d,(1,1)).view(Batch_size,-1)  #(B,C)
        global_avg_3d=F.adaptive_avg_pool2d(x3d_tfm,(1,1)).view(Batch_size,-1)  #(B,C)
        # print(global_avg_2d.size())
        # print(global_avg_3d.size())
        #mlp
        g=torch.cat((global_avg_2d,global_avg_3d),dim=1)
        # print(g.size())
        g=torch.relu(self.fc1(g))
        # print(g.size())
        g=torch.sigmoid(self.fc2(g))
        # print(g.size())
        g=g.view(Batch_size,channels_2d,1,1)
        # print(g.size())
        #融合2d,3d特征
        x2d_weight=x2d*g
        x3d_weight=x3d_tfm*g
        return x3d_weight+x2d_weight
    

class MDSU(nn.Module):  #多尺度反卷积上采样
    def __init__(self,input_channels,output_channels):
        super(MDSU,self).__init__()
        self.convs=nn.ModuleList([
            nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1),
            nn.Conv2d(input_channels,output_channels,kernel_size=5,padding=2),
            nn.Conv2d(input_channels,output_channels,kernel_size=7,padding=3),
            nn.Conv2d(input_channels,output_channels,kernel_size=9,padding=4),
            ])
        self.bn_relu=nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)   #节省内存
        )
        self.drop=nn.Dropout(p=0.3)
    def forward(self,x):
        features=[self.bn_relu(conv(x))  for conv in self.convs]
        concat_features=torch.cat(features,dim=1)
        return self.drop(concat_features)

class AMGR_NET(nn.Module):
    def __init__(self):
        super(AMGR_NET,self).__init__() ##(4,192,192)  (1,192,192,4)
        # 2D 编码器
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )      
        self.cpa1 = CPA(32, 192, 192)
        self.pool = nn.MaxPool2d(2)
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )          
        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  
        self.cpa2 = CPA(64, 96, 96)
        self.cff1 = CFF(64, 64) 
        self.conv2d_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  
        self.conv2d_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  
        self.cpa3 = CPA(128, 48, 48)
        self.cff2 = CFF(128, 128) 
        self.conv2d_7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  
        self.conv2d_8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  
        self.cpa4 = CPA(256, 24, 24)
        self.conv2d_9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  
        self.conv2d_10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  
        self.cpa5 = CPA(512, 12, 12)
        # 3D 编码器
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        ) 
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )    
        self.pool_1=nn.MaxPool3d(2)
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        ) 
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )          
        self.pool_2=nn.MaxPool3d(2)
        self.conv3d_5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        ) 
        self.conv3d_6 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        ) 
        # 解码器
        self.mdu1=MDSU()

    def forward(self, x2d, x3d):
        c1=self.conv2d_2(self.conv2d_1(x2d))
        
        
# if __name__ == '__main__':
#     # 参数设置
#     in_channels_2d = 64
#     in_channels_3d = 64               
#     pool_size = 8
#     decay_rate = 2

#     # 创建模块实例                                        
#     cff_module = CFF(in_channels_2d, in_channels_3d, pool_size, decay_rate)

#     # 创建测试输入数据
#     x2d = torch.randn(1, in_channels_2d, 32, 32)  # 2D特征图，形状为 (B, C2, H, W)
#     x3d = torch.randn(1, in_channels_3d, 32, 32, 16)  # 3D特征图，形状为 (B, C3, H, W, D)

#     # 前向传播
#     output = cff_module(x2d, x3d)
#     print(output.size())
#     # 创建MDSU模块
#     mdsu = MDSU(64, 64)

#     # 多尺度反卷积上采样
#     output_mdsu = mdsu(output)
#     print(output_mdsu.size())