import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
#import torch.multiprocessing as mp
# from diffusers.examples.research_projects.intel_opts.inference_bf16 import device


def make_coord(shape, ranges=None, flatten=True, align_corners=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)

        if align_corners:
            seq = v0 + (2 * r) * torch.arange(n + 1).float()
        else:
            seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_cell(bs, h, w):
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    coord = coord.repeat((bs,) + (1,) * coord.dim())
    cell = cell.repeat((bs,) + (1,) * cell.dim())
    return coord, cell


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    b, _, h, w = img.shape
    coord, cell = make_coord_cell(b, h, w)
    rgb = img.view(3, -1).permute(1, 0)
    return coord, cell, rgb

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        self.in_dim = in_dim
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class LIIF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list=[256, 256, 256, 256], cell_decode=True, local_ensemble=True):
        super().__init__()
        self.cell_decode = cell_decode
        self.local_ensemble = local_ensemble

        in_dim += 2  # attach coord
        if self.cell_decode:
            in_dim += 2
        self.imnet = MLP(in_dim, out_dim, hidden_list)

    def query_rgb(self, feat, coord, cell):
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0];
            areas[0] = areas[3];
            areas[3] = t
            t = areas[1];
            areas[1] = areas[2];
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def batched_predict(self, feat, coord, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(feat, coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    # b_size的作用是是否将数据分割成小批次处理  每个分区的大小是bsize
    def forward(self, feat, coord=None, cell=None, output_size=None, return_img=True, bsize=65536):
        if return_img:
            assert output_size is not None

        if self.training:
            bsize = 0

        if coord is None:
            assert output_size is not None
            coord, cell = make_coord_cell(feat.shape[0], output_size[0], output_size[1])
        if cell is None:
            assert output_size is not None
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / output_size[0]
            cell[:, 1] *= 2 / output_size[1]

        if bsize > 0:
            out = self.batched_predict(feat, coord, cell, bsize)
        else:
            out = self.query_rgb(feat, coord, cell)

        if return_img:
            out = rearrange(out, 'b (h w) c -> b c h w', h=output_size[0], w=output_size[1])

        return out






def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, scale_ratio=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)



class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.out_dim = block_in


    def forward(self, z, scale_ratio=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, scale_ratio)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, scale_ratio)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, scale_ratio)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

        # end
        return h

ddconfig={
  "double_z": False,
  "z_channels": 4,
  "resolution": 256,
  "in_channels": 4,
  "out_ch": 3,
  "ch": 128,
  "ch_mult": [ 1,2,4 ],  # num_down = len(ch_mult)-1
  "num_res_blocks": 2,
  "attn_resolutions": [],
  "dropout": 0.0,
}

liifconfig={
  "out_dim":3,
  "hidden_list": [256, 256, 256, 256],
  "cell_decode": True,
  "local_ensemble": True,
}
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.models.modeling_utils import ModelMixin
class IND(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    def __init__(self, ddconfig=ddconfig, liifconfig=liifconfig):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1),
            Decoder(**ddconfig)
        )
        self.inr = LIIF(in_dim=ddconfig['ch']*ddconfig['ch_mult'][0], **liifconfig)

    def forward(self, z, coord=None, cell=None, output_size=None, return_img=True, bsize=0):
        h = self.decoder(z)
        return self.inr(h, coord=coord, cell=cell, output_size=output_size,return_img=return_img,bsize=bsize)

# if __name__ =="__main__":
#     self.conv_out_conv = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
#     # 转成 LIIF
#     self.conv_out = LIIF(in_dim=block_out_channels[0], out_dim=out_channels, hidden_list=[256, 256, 256, 256])
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # UNet
    sample = torch.randn(1,256,120,160).to(device)
    coord = torch.randn(1,19200,2).to(device)
    cell = torch.randn(1,19200,2).to(device)
    shape = (120,160)
    return_img = True
    bsize=0
    in_dim = 256
    out_channels = 4
    conv_out_conv = nn.Conv2d(in_dim, out_channels, 3, padding=1).to(device)
    # 转成 LIIF
    conv_out = LIIF(in_dim=in_dim, out_dim=out_channels, hidden_list=[256, 256, 256, 256]).to(device)

    # 比较 参数量和计算量
    # 使用thop分析模型的运算量和参数量
    from thop import profile, clever_format
    MACs, params = profile(conv_out_conv, inputs=(sample,))

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.06f')

    print(f"UNet ConvOut sample model: 运算量：{MACs}, 参数量：{params}")
    # 原始的
    # sample = self.conv_outConv(sample)
    # 现在的
    # b_in, c_in, h_in, w_in = sample.shape
    # h_in, w_in = lq_h, lq_w
    h_new = 120
    w_new = 160
    b_in = 1
    # gt = torch.randn(b_in, c_in, h_new, w_new)
    coord, cell = make_coord_cell(b_in, h_new, w_new)
    # coord, cell, gt_rgb = to_pixel_samples(gt)
    return_img = True
    # b, c, h, w = gt.shape
    # t1 = time.time()
    # bsize是将数据切块大小 b=0表示不切块
    MACs, params = profile(conv_out,
                           inputs=(sample, coord, cell, [h_new, w_new], return_img, 0,))  # 估计参数大小

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.06f')

    print(f"UNet sample_out_LIIF model: 运算量：{MACs}, 参数量：{params}")
    # self, feat, coord=None, cell=None, output_size=None, return_img=True, bsize=65536
    # sample_out_LIIF = self.conv_out(sample, coord, cell, output_size=(h_new, w_new), return_img=return_img,
    #                                 bsize=0)
    ccc=0



    # VAE
    sample = torch.randn(1,128,480,640).to(device)
    coord = torch.randn(1,307200,2).to(device)
    cell = torch.randn(1,307200,2).to(device)
    shape = (480,640)
    return_img = True
    bsize=0
    in_dim = 128
    out_channels = 3
    conv_out_conv = nn.Conv2d(in_dim, out_channels, 3, padding=1).to(device)
    # 转成 LIIF
    conv_out = LIIF(in_dim=in_dim, out_dim=out_channels, hidden_list=[256, 256, 256, 256]).to(device)

    # 比较 参数量和计算量
    # 使用thop分析模型的运算量和参数量
    from thop import profile, clever_format
    MACs, params = profile(conv_out_conv, inputs=(sample,))

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.06f')

    print(f"VAE ConvOut sample model: 运算量：{MACs}, 参数量：{params}")
    # 原始的
    # sample = self.conv_outConv(sample)
    # 现在的
    # b_in, c_in, h_in, w_in = sample.shape
    # h_in, w_in = lq_h, lq_w
    h_new = 480
    w_new = 640
    b_in = 1
    # gt = torch.randn(b_in, c_in, h_new, w_new)
    coord, cell = make_coord_cell(b_in, h_new, w_new)
    # coord, cell, gt_rgb = to_pixel_samples(gt)
    return_img = True
    # b, c, h, w = gt.shape
    # t1 = time.time()
    # bsize是将数据切块大小 b=0表示不切块
    MACs, params = profile(conv_out,
                           inputs=(sample, coord, cell, [h_new, w_new], return_img, 0,))  # 估计参数大小

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.06f')

    print(f"VAE sample_out_LIIF model: 运算量：{MACs}, 参数量：{params}")
    # self, feat, coord=None, cell=None, output_size=None, return_img=True, bsize=65536
    # sample_out_LIIF = self.conv_out(sample, coord, cell, output_size=(h_new, w_new), return_img=return_img,
    #                                 bsize=0)
    ccc=0

    # net = IND(ddconfig,liifconfig).to(device)
    # a = torch.randn(5,4,64,64).to(device)
    # gt = torch.randn(5,3,256,256).to(device)
    # coord, cell, gt_rgb = to_pixel_samples(gt)
    # return_img = True
    # b,c,h,w = gt.shape
    # c = net(a, coord=coord, cell=cell, output_size=(h,w), return_img=return_img, bsize=b)
    # print(f"cc:{c.shape}")
