from torch import nn
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.modules.dfm import DF_Module
from basicsr.archs.keep_arch import CrossFrameFusionLayer
from diffusers.modules.main_blocks import double_depthwise_convblock
from diffusers.modules.common import FreBlock9,Attention

class XStartReview(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    def __init__(self,in_channels=4,out_channels=4,embedding_dim=256):
        super().__init__()
        self.q = double_depthwise_convblock(in_channels,
                                 embedding_dim,
                                embedding_dim)
        self.k = double_depthwise_convblock(in_channels,
                                 embedding_dim,
                                 embedding_dim)
        self.cross_frame = CrossFrameFusionLayer(
                dim=embedding_dim,  # 特征通道数
                num_attention_heads=2,  # 注意力头数量
                attention_head_dim=128,  # 每个注意力头的维度
                dropout=0.1,  # Dropout 概率
                cross_attention_dim=embedding_dim  # 跨帧注意力的维度
        )
        self.dfnet=DF_Module(embedding_dim,embedding_dim)
        self.out = double_depthwise_convblock(embedding_dim,
                                 embedding_dim,
                                 out_channels)

    def forward(self,x0,x0old):
        x0 = self.q(x0)
        x0old = self.k(x0old)
        x0old = self.cross_frame(x0old,x0)
        xout = self.dfnet(x0,x0old)
        return self.out(xout)

class XStartReviewDF(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    def __init__(self,in_channels=4,out_channels=4):
        super().__init__()
        self.net=DF_Module(in_channels,out_channels)
    def forward(self,x0,x0old):
        return self.net(x0,x0old)

class ConvFFT(nn.Module):
    def __init__(self, in_channels=4,out_channels=4,embedding_dim=256,num_embedding=1000):
        super(ConvFFT, self).__init__()
        # self.gamma = nn.Parameter(torch.ones(1),requires_grad=True)
        self.pre = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)
        self.cur = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)
        self.pre_att = Attention(dim=embedding_dim)
        self.cur_att = Attention(dim=embedding_dim)
        self.fuse = nn.Sequential(nn.Conv2d(2*embedding_dim, embedding_dim, 3, 1, 1), nn.Conv2d(embedding_dim, out_channels, 3, 1, 1), nn.Sigmoid())
        # self.fft = FreBlock9(channels=channels)
        self.time_embedding=nn.Embedding(num_embedding,embedding_dim)
        self.gammaLinear = nn.Linear(embedding_dim,in_channels)
        # self.weight = nn.Linear(pool_size*pool_size,2)
        # self.pool_size = pool_size
    def forward(self, input_pre, input_cur,timestep=None):
        if input_pre is None:
            return input_cur
        #转换到频率域
        # emb = self.time_embedding(timestep[0])          #前一个
        b,c,_,_ = input_pre.shape
        emb_cur = self.time_embedding(timestep[1])          #后一个
        gamma = self.gammaLinear(emb_cur).unsqueeze(-1).unsqueeze(-1)                  # gamma 是一个和时间有关系的数
        # input_pre_frq = self.fft(input_pre)
        # input_cur_frq = self.fft(input_cur)
        input_pre_frq = input_pre
        input_cur_frq = input_cur
        # # ori = spa
        # pre = self.pre(input_pre_frq) + emb.unsqueeze(-1).unsqueeze(-1)
        pre = self.pre(input_pre_frq)
        # pre = input_pre_frq
        # cur = self.cur(input_cur_frq) + emb_cur.unsqueeze(-1).unsqueeze(-1)
        cur = self.cur(input_cur_frq)
        # cur = input_cur_frq
        pre = self.pre_att(pre, cur) + pre
        cur = self.cur_att(cur, pre) + cur
        weight = self.fuse(torch.cat((pre, cur), 1))
        # adaverage_pool = nn.AdaptiveAvgPool2d(output_size=(self.pool_size, self.pool_size))  # 输出大小的尺寸指定为100*100
        # update_gate = adaverage_pool(fuse)
        # update_gate = rearrange(update_gate,"b c h w -> b c (h w)")
        # param = self.weight(fuse)
        # wei, bias = fuse.chunk(2, dim=1)
        # wei = rearrange(wei,"b c (h w) -> b c h w",h=1,w=1)
        # bias = rearrange(bias,"b c (h w) -> b c h w",h=1,w=1)
        #b c h w
        # fuse = torch.sigmoid(fuse)
        h_next = (1 - gamma * weight) * input_cur + gamma * weight * input_pre
        h_next = torch.nan_to_num(h_next, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return h_next

class ConvFFTFreq(nn.Module):
    def __init__(self, in_channels=4,out_channels=4,embedding_dim=256,num_embedding=1000):
        super(ConvFFTFreq, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1),requires_grad=True)
        self.pre = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)
        self.cur = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)
        self.pre_att = Attention(dim=embedding_dim)
        self.cur_att = Attention(dim=embedding_dim)
        self.fuse = nn.Sequential(nn.Conv2d(2*embedding_dim, in_channels, 3, 1, 1), nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.Sigmoid())
        # self.fft = FreBlock9(channels=in_channels)
        self.time_embedding=nn.Embedding(num_embedding,embedding_dim)
        # self.weight = nn.Linear(pool_size*pool_size,2)
        # self.pool_size = pool_size
    def forward(self, input_pre, input_cur,timestep=None):
        if input_pre is None:
            return input_cur
        #转换到频率域
        emb = self.time_embedding(timestep)
        # emb = self.time_embedding(timestep)
        # input_pre_frq = self.fft(input_pre) + emb.unsqueeze(-1).unsqueeze(-1)
        # input_cur_frq = self.fft(input_cur)
        input_pre_frq = input_pre
        input_cur_frq = input_cur
        # # ori = spa
        pre = self.pre(input_pre_frq) + emb.unsqueeze(-1).unsqueeze(-1)
        # pre = self.pre(input_pre_frq)
        cur = self.cur(input_cur_frq)
        pre = self.pre_att(pre, cur) + pre
        cur = self.cur_att(cur, pre)+cur
        weight = self.fuse(torch.cat((pre, cur), 1))
        # adaverage_pool = nn.AdaptiveAvgPool2d(output_size=(self.pool_size, self.pool_size))  # 输出大小的尺寸指定为100*100
        # update_gate = adaverage_pool(fuse)
        # update_gate = rearrange(update_gate,"b c h w -> b c (h w)")
        # param = self.weight(fuse)
        # wei, bias = fuse.chunk(2, dim=1)
        # wei = rearrange(wei,"b c (h w) -> b c h w",h=1,w=1)
        # bias = rearrange(bias,"b c (h w) -> b c h w",h=1,w=1)
        #b c h w
        # fuse = torch.sigmoid(fuse)
        h_next = (1 - self.gamma * weight) * input_cur + self.gamma * weight * input_pre
        h_next = torch.nan_to_num(h_next, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return h_next

class XStartReviewCrossFFT(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    def __init__(self,in_channels=4,out_channels=4):
        super().__init__()
        self.net=ConvFFT(in_channels,out_channels)
    def forward(self,x0,x0old,timestep=None):
        return self.net(x0,x0old,timestep)

class XStartReviewCrossFreq(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    def __init__(self,in_channels=4,out_channels=4):
        super().__init__()
        self.net=ConvFFTFreq(in_channels,out_channels)
    def forward(self,x0,x0old,timestep=None):
        return self.net(x0,x0old,timestep)