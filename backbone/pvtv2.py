import math
from functools import partial

import torch
import torch.nn as nn
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron, MLP 多层感知器
    :in_features: 输入尺寸
    :hidden_features: 隐藏层尺寸
    :out_features: 输出尺寸
    :act_layer: 激活(函数)层，此处默认使用 GELU
    :drop: 随机丢弃使用的概率 p, 默认值 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 输出尺寸 , or的意思是若 out_features 存在则用之，反之则用 in_features
        out_features = out_features or in_features
        # 隐藏层尺寸 , or的意思是若 hidden_features 存在则用之，反之则用 in_features
        hidden_features = hidden_features or in_features

        # 全连接层1，一个线性层输入尺寸in_features, 输出尺寸 hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数是 GELU
        self.act = act_layer()
        # 深度可分离卷积，hidden_features 作为卷积的输入输出通道数
        self.dwconv = DWConv(hidden_features)
        # 全连接层2，一个线性层输入通道数 hidden_features, 输出通道数 out_features
        self.fc2 = nn.Linear(hidden_features, out_features)

        # 将输入 Tensor 内的元素，按照概率 drop 做随机丢弃，相当于正则化，防止过拟合
        self.drop = nn.Dropout(drop)
        # 使用 self.apply(fn) 方法递归的将权重初始化方法作用于 Mlp 模块的每一个子模块
        self.apply(_init_weights)

    def forward(self, x, H, W):
        """
        首先 forward 函数的输入是 Attention 的输出和原始输入残差相加的结果，
        输入大小是 (1, 3136, 64)
        fc1 输出 (1, 3136, 512)
        act 是 GELU 激活函数, 输出 (1, 3136, 512)
        drop 输出 (1, 3136, 512)
        fc2 输出 (1, 3136, 64)
        drop 输出 (1, 3136, 64)
        """
        x = self.fc1(x)  # 将输入 Tensor x 传入 fc1
        x = self.dwconv(x, H, W)  # 将经过 fc1 处理后的 Tensor x 传入 DWConv 进行前向传播
        x = self.act(x)  # 将经过 DWConv 处理后的 Tensor x 传入激活函数 GELU
        x = self.drop(x)  # 将经过 GELU 处理后的 Tensor x 传入 dropout 层进行处理
        x = self.fc2(x)  # 将经过 dropout 层处理后的 Tensor x 传入 fc2
        x = self.drop(x)  # 将经过 fc2 处理之后的 Tensor x 再传入 dropout
        return x


class Attention(nn.Module):
    """
    Linear Spatial Reduction Attention LSRA 线性空间缩减注意力
    :dim: 输入 token 维度
    :num_heads: 注意力多头数量
    :qkv_bias: 是否使用偏置，默认值 False
    :qk_scale: 缩放因子，默认值 None
    :attn_drop: 注意力比例
    :proj_drop: 映射比例
    :sr_ratio: 维度缩减率
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # 维度 dim 必须可以整除 num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # 输入维度 dim
        self.num_heads = num_heads  # 注意力头数量
        head_dim = dim // num_heads  # 计算每一个注意力头的输入维度
        self.scale = qk_scale or head_dim ** -0.5  # 得到根号下d_k分之一的值

        # 一个全连接层输入尺寸=输出尺寸=dim, 是否使用偏置是 qkv_bias, 用于生成 q
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # 一个全连接层输入尺寸是 dim, 输出尺寸是 2*dim , 是否使用偏置是 qkv_bias, 用于生成 k, v
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # 一个随机丢弃层

        self.proj = nn.Linear(dim, dim)  # 一个全连接层输入尺寸=输出尺寸=dim, 偏置默认 True
        self.proj_drop = nn.Dropout(proj_drop)  # 一个随机丢弃层

        self.sr_ratio = sr_ratio  # 维度缩减率 R，sr_ratios = [8, 4, 2, 1], 对应不同阶段的缩减大小
        if sr_ratio > 1:  # 若缩减率大于 1, sr层是一个卷积层，输入通道数 = 输出通道数 = dim, 卷积核大小 = 步长 = sr_ratio
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # 相当于下采样
            self.norm = nn.LayerNorm(dim)
        # 使用 self.apply(fn) 方法递归的将权重初始化方法作用于 Attention 模块的每一个子模块
        self.apply(_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape  # Tensor x 的 shape 是 (B, N, C), 其中 N = H * W，C 代表总嵌入维度 total_embed_dim
        # B,N,T,Tc => B,T,N,Tc
        # x.permute(0, 2, 1, 3) 即将 Tensor x 的第二，三个维度进行交换
        # Tensor x 传入全连接层 q, 输出还是 (B, N, C) 因为入出 dim 一样,
        # 然后将输出尺寸调整为 (B, N, num_heads, channel_per_heads), 最后调整为 (B, num_heads, N, channel_per_heads)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:  # 带缩减率的 SRA
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # 将输入 Tensor x 的 N 和  C 互换(B, N, C->B,C,N), 然后重塑维度为 (B,C,H,W)
            # 将上一步输出的 Tensor x_ 输入下采样卷积 sr, 采样完毕输出的结果是 (B, C, H/sr_ratio, W/sr_ratio), 然后重塑维度为 (B, C, HW/sr_ratio^2)
            # 再将输出 Tensor 的 后两个维度互换，即变为 (B, HW/sr_ratio^2, C)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)  # 进行层归一化处理, shape 保持不变
            # 这一行就是和 Vit 一样用 x 生成 k 和 v，不同的是这里的 x 通过卷积的方式降低了 x 的大小。
            # 这一行 shape 的变化 => (B, HW/sr_ratio^2, C) -> (B, HW/sr_ratio^2, 2*C)
            # -> (B, HW*num_heads/sr_ratio^2, 2, 1, 2*C/num_heads) -> (2, B, 1, HW*num_heads/sr_ratio^2, 2*C/num_heads)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:  # 不带缩减率的 SRA
            # -> B, QK, N, H, C => QK, B, T, N, Tc
            # Tensor x 的维度变化 => (B, N, C) -> (B, N, 2*C)
            # -> (B, N', 2, num_heads, C/num_heads) -> (2, B, num_heads, N', C/num_heads)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 拿到 kv:(2, B, 1, HW*num_heads/sr_ratio^2, 2*C/num_heads)分别取 index 为 0 和 1
        # 即可得到 k 和 v, 即: k, v = kv[0], kv[1] , 因此 k 和 v 的 shape 为 (B, 1, HW*num_heads/sr_ratio^2, 2*C/num_heads)
        k, v = kv[0], kv[1]

        # q 和 k对后两个维度转置的矩阵相乘，再乘以 scale
        # q @ (B, 1, 2*C/num_heads, HW*num_heads/sr_ratio^2) = (B, num_heads, num_patches+1, num_patches+1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 对 attn 做 softmax激活，按照最后一个维度，即进行激活
        attn = self.attn_drop(attn)  # 对激活之后的 attn 做随机丢弃

        # attn 与 v 做矩阵乘法 (
        # (B, num_heads, num_patches+1, num_patches+1) x (B, num_heads, num_patches+1, channel_per_heads)
        # = (B, num_heads, num_patches+1, channel_per_heads)) 然后将结果 Tensor 的第二三个维度转置，最后重塑为 (B. N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 将 Tensor x 输入全连接层 proj
        x = self.proj_drop(x)  # 将 Tensor x 输入随机丢弃层 proj_drop
        return x


class Block(nn.Module):
    """
    包含整个 Transformer Encoder 的所有结构，是 Transformer Encoder 的主要组件，
    其内部执行了两次残差融合（原始特征图 x 和 Attention 处理后的特征图，Attention 处理后的特征图和 MLP 处理后的特征图
    :dim: 输入维度，输入节点数目
    :num_heads: 注意力头数量
    :mlp_ratio: 多层感知器中隐藏层节点数目与输入节点数目的比例
    :qkv_bias: query, key, value 向量是否使用偏置，默认值 False
    :qk_scale: query * key 的缩放因子，默认值 None
    :drop: 随机丢弃使用的概率 p, 默认值  0.0
    :attn_drop: 注意力比例
    :drop_path: 随机丢弃路径概率 p, 默认值 0.0
    :act_layer: 激活函数，这里使用的是 GELU
    :norm_layer: 归一化层，这里使用的是层归一化 LayerNorm
    :sr_ratio: 缩减率
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # 归一化层 1
        self.norm1 = norm_layer(dim)
        # 实例化一个注意力模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 这里做了一个三元运算，即若丢弃路径概率大于 0.0 则进行丢弃路径操作，否则直接返回 drop_path=0.0
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 归一化层 2
        self.norm2 = norm_layer(dim)
        # 多层感知器的隐藏层尺寸为 输入维度 乘以 多层感知器中隐藏层节点数目与输入节点数目的比例
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 实例化一个多层感知器模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        # 从输入 Tensor x 中拷贝一份，留给残差结构使用。
        # 然后输入的 x 先经过一层 LayerNorm 层，维度不变，再经过  SRA 层 (SR + MHA)与之前拷贝的输入进行叠加
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 经过 SRA 层的特征图拷贝一份，留给残差结构使用。
        # 然后将经过 LayerNorm 层，维度不变。再送入 FeedForward 层，之后与之前拷贝的输入叠加
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding 切图重排，对原始输入图像做切块处理
    编码图像块特征信息，得到特征响应图和当前特征图的长宽
    :img_size: 图片大小，默认值 224
    :patch_size: 切块的大小，默认值 7，进而可知 patch_num = 224/7=32
    :stride: 步长，默认值 4
    :in_chans: 输入图像通道数，默认值 3
    :embed_dim: 嵌入维度，亦即输出的每一个 token 的维度，默认值 768 = 16*16*3，即 (H*W*C)
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        # 将图片尺寸转成一个长度为 2 的元组，默认是 224=>(224, 224)
        img_size = to_2tuple(img_size)
        # 将切块尺寸转成一个长度为 2 的元组，默认是 7=>(7, 7)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # 将图片的高整除切块的高作为 H, 图片的宽整除切块的宽作为 W，即横向和纵向的数量
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # 求出切块的数量 = 横向数量*纵向数量
        self.num_patches = self.H * self.W

        # 通过卷积实现 OPE 只需要调整卷积核的大小和卷积步长即可将输入图像切成多个小块
        # 一个输入通道数 in_chans, 输出通道数 embed_dim, 大小为 patch_size[0] * patch_size[1],
        # 步长 stride, 填充 (patch_size[0]/2, patch_size[1]/2) 的卷积 => 这样就可以实现将输入图像尺寸切割成大小为 kernel_size^2 的多个小 patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        # 使用 self.apply(fn) 方法递归的将权重初始化方法作用于 OverlapPatchEmbed 模块的每一个子模块
        self.apply(_init_weights)

    def forward(self, x):
        # 使用卷积实现图像切块操作
        x = self.proj(x)
        # Tensor x 的高和宽,  _表示占位, 为了避免取错
        _, _, H, W = x.shape
        # 将 Tensor x 从第三个维度开始展平然后将第 2, 3个维度互换
        # 即 B, C, H, W => B, C, (H * W) => B, (H*W), C => B, N, C
        x = x.flatten(2).transpose(1, 2)
        # 将 Tensor x 传入层标准化进行处理
        x = self.norm(x)
        # 返回 Tensor x, 高 和 宽
        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    """
    PyramidVisionTransformer => PVT v2
    :img_size: 图片尺寸，默认值 224
    :patch_size: 块大小，默认值 16
    :in_chans: 图片通道数
    :num_classes: 分类任务类型数量，默认值 1000
    :embed_dims: 嵌入维度，默认值 [64, 128, 256, 512]
    :num_heads: Multi-head Attention Layer的 head 数，默认值分别为 [1, 2, 4, 8]
    :mlp_ratios:多层感知器比例列表，默认值 [4, 4, 4, 4]
    :qkv_bias: 是否使用偏置，默认值 False
    :qk_scale: 缩放因子，默认值 None
    :drop_rate: 随机丢弃使用的概率 p, 默认值 0.0
    :attn_drop_rate: 注意力随机丢弃概率，默认值 0.0
    :drop_path_rate: 随机丢弃路径概率 p, 默认值 0.0
    :norm_layer: 归一化层，使用的是 LayerNormalization
    :depths: depths[i] => 第 i 个 stage 有多少个 Block
    :sr_ratios: 缩减率，默认值分别为 [8, 4, 2, 1]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes  # 我是分割任务，所以这个参数暂时可有可无
        self.depths = depths

        # patch_embed => Patch Embedding 用于编码图像信息
        # 创建一个 重叠分块嵌入 模块，图像尺寸是 img_size, 块大小 7, 步长 4, 输入通道数为 in_chans, 嵌入维度为 embed_dim[0]
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder => Transformer Encoder 用于编码和丰富特征信息
        # dpr = drop_path_rate 一个列表，用于存储每个 Block层 的随机丢弃路径概率值
        # 这里取了 16 个概率值，x.item() 用于将张量值 转为 标量值
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        cur = 0  # 当前下标
        # 将 depth[0] 个 Block 按照顺序添加到 ModuleList 容器中。比如，这里 block1 是取 0,1,2三个下标对应的概率值，分别赋值给三个 Block
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head, here is not about classification task but segmentation task
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(_init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]

        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore  # 这个方法设置忽略并保留为 Python 函数
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.forward_features(x)

        # x 即上面方法里的返回值 outs 即一个包含4个不同尺寸Tensor的列表
        return x

    def forward_features(self, x):  # x => [1, 3, 352, 352]
        B = x.shape[0]  # 取输入 Tensor x 的 batch_size
        outs = []  # 一个列表，存入 stage 1 - 4 的输出 Tensor

        # stage 1
        x, H, W = self.patch_embed1(x)  # patch_embed 操作(切片)
        for i, blk in enumerate(self.block1):  # stage 1有  depth[0] 个 Transformer Block, 所以枚举遍历
            x = blk(x, H, W)  # 进入当前的 Transformer Block 进行处理
        x = self.norm1(x)  # 进入 LayerNorm 1 进行处理
        # 将 Tensor x 重塑成 B*H*W*C , -1 表示没写的维度 channel 经过其他三个维度推断得出,
        # 然后将最后一个维度也就是 C 换到第二个维度上，即 B*C*H*W, 并使得新的 Tensor在内存中保持连续
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 将经过处理之后的输出 Tensor x 存入 outs 中
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class DWConv(nn.Module):
    """
    Depth-wise Separable Convolution(深度可分离卷积) => DWConv
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # 创建一个输入通道数 dim, 输出通道数 dim, 大小 3*3，步长 1, 填充 1, 使用偏置, 从输入通道到输出通道的阻塞连接数 dim
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape  # 取输入 Tensor 的 batch_size, height*width, channel
        x = x.transpose(1, 2).view(B, C, H, W)  # 将输入 Tensor 的 后两个维度(N & C)互换，然后重组为 B*C*H*W
        x = self.dwconv(x)  # 输入 Tensor x 经过 dwconv 处理
        # 将从第三个维度开始的后两个维度进行压缩，压缩之后变成 B*C*(H*W) = B,C,N, 然后将后两个维度互换变成 B*(H*W)*C = B,N,C
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def _init_weights(m):  # 初始化权重
    if isinstance(m, nn.Linear):  # 若当前模块是线性层
        # 截断当前模块权重的正态分布，将权重调整为正态分布，标准差为0.02
        # 替代的写法是 nn.init.trunc_normal_(m.weight, std=.02)
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:  # 若当前模块的偏置不为空
            nn.init.constant_(m.bias, 0)  # 使用 0 填充模块的 bias
    elif isinstance(m, nn.LayerNorm):  # 若当前模块是 LayerNorm 层
        nn.init.constant_(m.bias, 0)  # 使用 0 填充模块的 bias
        nn.init.constant_(m.weight, 1.0)  # 使用 1.0 填充模块的 weight
    elif isinstance(m, nn.Conv2d):  # 若当前模块是卷积
        # 计算当前卷积层的扇出，即卷积尺寸相乘再乘以输出通道数
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups  # 计算平均扇出
        # 将模块的权重缩放成均值为 0，方差为 根号(2.0/fan_out) 的状态
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:  # 若当前模块的偏置不为空
            m.bias.data.zero_()  # 使用 0 填充模块的 bias


@register_model
class PvtV2B2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(PvtV2B2, self).__init__(  # 初始化超类 PyramidVisionTransformerImpr
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
