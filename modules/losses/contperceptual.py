import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

#变分自编码器的损失函数
'''
该类用于计算VAE的损失 
损失由四部分组成 
    真实图-生成图 像素级别的L1损失
    真实图-生成图 特征级别的相似度损失
    VAE的KL损失
    生成器和鉴别器的损失
__init__:
    disc_start. 用于开始应用鉴别器损失的迭代次数, 影响GAN损失的权重
    logvar_init. 对数方差的初始值, 用于衡量重构损失和正则损失
    kl_weight. KL损失的权重. (KL损失: VAE的预测高斯分布和标准高斯分布的KL损失, 这一损失也被认为是VAE中的一个正则损失
    pixelloss_weight. 像素损失的权重. 但这个参数在代码中完全没有用到. (像素损失: 真实的图像和生成的图像之间的L1损失
    disc_weight. 生成器/鉴别器损失的权重. (生成/鉴别损失: 对于鉴别器, 要识别真实图像/生成图像; 对于生成器, 要欺骗鉴别起). 这一参数和上面的 disc_start 共同影响GAN损失的权重
    perceptual_weight. 感知相似损失的权重. (感知相似损失: 和像素损失类似, 保证真实图像和生成图像相似. 感知损失是把图像放入VGG中, 计算各层的特征, 并计算特征之间的相似性)
    disc_num_layers. 鉴别器的层数
    disc_in_channels. 鉴别器的输入通道数
    disc_factor. 控制GAN损失的因子. 它和上面的 disc_start, disc_weight 共同最终决定GAN损失的权重.
    use_actnorm. 是否在GAN中使用激活归一化 (ActNorm)
    disc_conditional. 鉴别器是否为有条件的
    disc_loss. 鉴别器损失函数的类型         
    disc_start: 50001
    kl_weight: 0.000001
    disc_weight: 0.5
'''
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start=0.0, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        #用于计算两个图像的感知相似度
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
    #计算自适应权重 来平衡真实图和生成图的损失和生成/鉴别的损失
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    #前向过程 计算损失
    '''
        input  真实的输入图像
        reconstructions. VAE重构的图像
        posteriors. VAE中间层预测的均值和方差的分布.
        optimizer_idx. 一个指示器, 当其为 0 时优化生成器, 1 时优化鉴别器.
    '''
    # def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
    #             global_step, last_layer=None, cond=None, split="train",
    #             weights=None):
    def forward(self, inputs, reconstructions, optimizer_idx,
                    global_step, last_layer=None, cond=None, split="train",
                    weights=None):
        #rec_loss为原图和生成图的L1距离
        b,c,h,w = inputs.shape
        pix_sum = b * c * h * w
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            #p_loss是LPIPS损失，由图像的每一层vgg特征之间的相似度计算得来
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            #乘一个因子 self.perceptual_weight来衡量不同损失的重要程度
            #重构损失=L1距离+w*LPIPS损失
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        #计算非负对数似然
        #self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        #这是一个可学习的数. 通过将重构误差 rec_loss 正则化为 nll_loss,
        #允许模型估计重构误差的不确定性. 通过这种方式, 模型可以学习在哪些区域的重构更加困难.
        #例如, 如果模型认为某个区域的重构更加困难, 可以通过增加该区域的 self.logvar 值来降低重构误差的影响,
        # 这有助于模型更加健壮, 更好地应对有噪声的数据.

        #为什么要在后面加上 self.logvar ? 这其实也很容易理解, 我们不希望模型无脑地增加不确定性.
        #如果我们不加上 self.logvar, 那可能陷入一种这样的情况:
        #模型无限地增加 self.logvar, 认为重构总是很困难, 最终让重构误差 nll_loss 趋于 0,
        #并只考虑正则化误差. 这显然是不合适的, 因此在后面加上对数方差, 让模型能在两种情况下作出选择.
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # #计算后验分布和标准高斯分布之间的距离
        # kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        #下面的损失是用于训练GAN部分的
        #optimizer_idx有两个取值 0或1 0时更新生成器 1时更新鉴别器
        if optimizer_idx == 0:
            # generator update
            #更新生成器
            if cond is None:                        #cond表示是否有条件判断
                assert not self.disc_conditional    #无条件判断
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional        #有条件判断
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            #logits_fake是判别器的输出
            #注意输入的是 reconstructions，这是假数据，当前正在训练生成器 目标是欺骗鉴别器
            #鉴别器：真数据---> 0;  假数据 ---> 1
            g_loss = -torch.mean(logits_fake)                   #生成器损失

            #下面是给生成器损失乘以一个权重，目的是加强训练生成器
            #当生成器权重<=0.0时，不再使用生成器
            #生成器只在训练VAE阶段使用，在训练diffusion阶段不用
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
            #小于 iter_start' disc_factor=0.0
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            #损失 = 重构损失（weighted_nll_loss）+正则KL损失（kl_loss）+生成器损失（g_loss）
            # loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            loss = weighted_nll_loss  + d_weight * disc_factor * g_loss
            loss = loss / pix_sum
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            # log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
            #        "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
            #        "{}/rec_loss".format(split): rec_loss.detach().mean(),
            #        "{}/d_weight".format(split): d_weight.detach(),
            #        "{}/disc_factor".format(split): torch.tensor(disc_factor),
            #        "{}/g_loss".format(split): g_loss.detach().mean(),
            #        }
            return loss, log

        if optimizer_idx == 1:
            #更新鉴别器
            # second pass for discriminator update
            if cond is None:                #同上 是否有条件鉴别
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            #同上，鉴别器权重
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            #self.disc_loss给出了如何训练鉴别器
            #鉴别器损失 鉴别器的损失有两种
            #self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            d_loss = d_loss / pix_sum
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

#
class LPIPSWithMSE(nn.Module):
    def __init__(self,in_channels=3, disc_start=0.0, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        #用于计算两个图像的感知相似度
        self.perceptual_loss = LPIPS().eval()
        self.transform = nn.Conv2d(in_channels,3,1)
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.mse = nn.MSELoss()
        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm
        #                                          ).apply(weights_init)
        # self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional
    #计算自适应权重 来平衡真实图和生成图的损失和生成/鉴别的损失
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    #前向过程 计算损失
    '''
        input  真实的输入图像
        reconstructions. VAE重构的图像
        posteriors. VAE中间层预测的均值和方差的分布.
        optimizer_idx. 一个指示器, 当其为 0 时优化生成器, 1 时优化鉴别器.
    '''
    # def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
    #             global_step, last_layer=None, cond=None, split="train",
    #             weights=None):
    def forward(self, inputs, reconstructions,weights=None):
        #rec_loss为原图和生成图的L1距离
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.mse(inputs.contiguous(), reconstructions.contiguous())

        # #计算非负对数似然
        # #self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # #这是一个可学习的数. 通过将重构误差 rec_loss 正则化为 nll_loss,
        # #允许模型估计重构误差的不确定性. 通过这种方式, 模型可以学习在哪些区域的重构更加困难.
        # #例如, 如果模型认为某个区域的重构更加困难, 可以通过增加该区域的 self.logvar 值来降低重构误差的影响,
        # # 这有助于模型更加健壮, 更好地应对有噪声的数据.
        #
        # #为什么要在后面加上 self.logvar ? 这其实也很容易理解, 我们不希望模型无脑地增加不确定性.
        # #如果我们不加上 self.logvar, 那可能陷入一种这样的情况:
        # #模型无限地增加 self.logvar, 认为重构总是很困难, 最终让重构误差 nll_loss 趋于 0,
        # #并只考虑正则化误差. 这显然是不合适的, 因此在后面加上对数方差, 让模型能在两种情况下作出选择.
        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # weighted_nll_loss = nll_loss
        # if weights is not None:
        #     weighted_nll_loss = weights*nll_loss
        #
        # # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        loss_final = rec_loss
        if self.perceptual_weight > 0:
            input_trans = self.transform(inputs)
            target_trans = self.transform(reconstructions)
            #p_loss是LPIPS损失，由图像的每一层vgg特征之间的相似度计算得来
            p_loss = self.perceptual_loss(input_trans.contiguous(), target_trans.contiguous())
            #乘一个因子 self.perceptual_weight来衡量不同损失的重要程度
            #重构损失=L1距离+w*LPIPS损失
            loss_final = loss_final + self.perceptual_weight * p_loss
            loss_final = torch.sum(loss_final) / loss_final.shape[0]
            ccc=0

        return loss_final

