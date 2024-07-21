import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差块
class Bottle2neck(nn.Module):
    # 输出通道数的倍数
    expansion = 4

    ''' Constructor
    Args:
        inplanes:         input channel dimensionality
        planes:             output channel dimensionality
        stride:              conv stride. Replaces pooling layers，避免池化操作造成信息丢失
        downsample: None when stride = 1
        baseWidth:     basic width of conv 3x3，分组基本宽度
        scale:                 number of scale，论文中的 s，即组的数量。
        stype:                'normal': normal set. 'stage': first block of a new stage，stage即不需要合并其他输出的组，normal即需要合并其他输出的组
    '''

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        # 调用 Bottleneck 的父类 => Module，并初始化之
        super(Bottle2neck, self).__init__()
        # planes(输出通道维度) 乘以 baseWidth(每个组的基本宽度) 除以 64.0, 然后 [下取整] 转成整数类型即宽度 width，也就是每组分得的通道数
        width = int(math.floor(planes * (baseWidth / 64.0)))
        # conv1 由 inplanes(输入通道数量), width*scale(输出通道数量)
        # 应由 kernel_size=1 改为  kernel_size=(1,1) ，因为这里要的是一个 tuple 类型，另此处的 kernel_size 代表 conv1 是 1*1 的卷积
        # bias=False 表示不使用偏置
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        # bn1 是 conv1 之后的一个批量归一化层，传入输出通道 Channels = width * scale
        self.bn1 = nn.BatchNorm2d(width * scale)

        # 若分组数等于 1
        if scale == 1:
            # 将 nums 设为 1
            self.nums = 1
        else:
            # 将 nums 设为 scale - 1，-1是因为这一层后接 3*3 卷积的只有 scale - 1个组
            self.nums = scale - 1

        # 若当前为新阶段（层）的第一个块
        if stype == 'stage':
            # 一个大小 3*3，步长 stride，填充 1 的池化层
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        # convs 是一个卷积层列表，bns 是一个批量归一化层列表
        convs, bns = [], []
        # 需要 3*3 卷积的组的数量（这里使用的是 => 1, 2, 3组）
        for i in range(self.nums):
            # 将一个输入通道数为 width，输出通道数为 width，大小 3*3，步长 stride，填充 1，不使用 bias的卷积加入 convs
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            # 将一个 大小为 width 的 BN 层加入 bns
            bns.append(nn.BatchNorm2d(width))
        # 将 convs 添加到 ModuleList内，它可以将参数自动注册到网络上
        self.convs = nn.ModuleList(convs)
        # 将 bns 添加到 ModuleList 内，它可以将参数自动注册到网络上
        self.bns = nn.ModuleList(bns)

        # 一个输入通道数为 width*scale，输出通道数为 planes*expansion，大小 1*1，不使用 bias 的卷积
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        # 大小为 planes * expansion 的 BN 层
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # 给 downsample, stype, scale, width 赋值
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        # 这里先用 conv1 处理输入 x，然后传入 bn1 层，最后经过 RELU 层激活，此处使用了 inplace=True，就地修改不占用额外空间
        global sp
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # 将输出张量 out 按照 width 以及 通道数进行划分，每组通道数 26，分成 4 组 => [n, 104], 104//26 = 4，spx是一个输出张量
        spx = torch.split(out, self.width, 1)
        # 遍历需要后接 3*3 卷积的组数量 nums，这里是处理 0-2 这三个组
        for i in range(self.nums):
            # 第一个 i==0 需要完成对 sp 赋值，即 sp=spx[0]
            # 对于 stype=='stage' 的时候，不用加上前一小块的输出结果，而是直接 sp=sp[i]
            # 因为输入，输出的尺寸不一致（通道数不一致），所以无法加一起
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            # 将上一组的输出加入这一组，一起投入到这一组的 3*3 卷积里
            else:
                sp = sp + spx[i]
            # 当前组的 sp (批，高，宽固定，通道数 26) 经过它自己的 3*3 的卷积（这边处理的还真的是 1, 2, 3 组后接 3*3 卷积）
            sp = self.convs[i](sp)
            # 然后当前的张量 sp 经过当前的 bn 和 relu 处理（本地修改）
            sp = F.relu(self.bns[i](sp), inplace=True)
            # 若· i==0，直接将张量 sp 赋值给张量 out。
            # 不然的话，在维度 1 （通道数）上进行 out 和 sp 的拼接
            out = sp if i == 0 else torch.cat((out, sp), 1)

        # 最后一组 spx[self.nums] 不进行卷积全部拼接回原来的维度
        if self.scale != 1 and self.stype == 'normal':
            # 按照 通道数 拼接，将 out 和 最后一组进行拼接
            out = torch.cat((out, spx[self.nums]), 1)

        # 这里需要加上 pool 的原因是 => 对于每一个 layer 的 stage 模块，它的 stride 不是固定的，layer1 的 stride=1
        # 而 layer2, 3, 4 的 stride=2，前三小块都经过了 stride=2 的 3*3 卷积，而第四个小块是直接送到 y 中的，但它必须 pool 一下
        # 不然的话，尺寸和前三个小块对不上，无法完成最后的 concatenation 操作。
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # 1*1的卷积
        out = self.bn3(self.conv3(out))
        # 若当前输入张量 x需要下采样
        if self.downsample is not None:
            # 则输入张量  x 进行下采样
            x = self.downsample(x)
        # 将输出张量 out 和输入张量 x 进行原地加和，并投入 relu
        return F.relu(out + x, inplace=True)


class Res2Net(nn.Module):
    """初始化 Res2Net 类，传入 4 个参数
        layers:           代表 2-5 层含有的 Bottleneck 个数
        snapshot:     代表预训练模型参数保存文件和其位置
        baseWidth: 分组基础宽度
        scale:             代替 3*3 卷积的操作的分组数，这里是 Res2Net 的标配即 4 组
    """

    def __init__(self, layers, snapshot, baseWidth=26, scale=4):
        # 输入通道数为 64
        self.inplanes = 64
        # 调用 Res2Net 的父类 => Module，并初始化之
        super(Res2Net, self).__init__()
        # Res2Net的预训练模型参数保存文件及其位置
        self.snapshot = snapshot
        # Res2Net的分组基础宽度，这里默认是 26
        self.baseWidth = baseWidth
        # Res2Net的分组数
        self.scale = scale

        # Res2Net的第一个卷积，它是一个顺序容器
        self.conv1 = nn.Sequential(
            # 一个输入通道数为 3，输出通道数为 32的大小为 3*3，步长 2，填充 1，不使用 bias 的卷积
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            # BN层
            nn.BatchNorm2d(32),
            # RELU层，不创建新变量，在原变量基础上进行处理
            nn.ReLU(inplace=True),
            # 一个输入通道数为 32，输出通道数为 32，大小为 3*3，步长 2，填充 1，不使用 bias 的卷积
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            # BN层
            nn.BatchNorm2d(32),
            # RELU层，不创建新变量，在原变量基础上进行处理
            nn.ReLU(inplace=True),
            # 一个输入通道数为 32，输出通道数为 64，大小为 3*3，步长 2，填充 1，不使用 bias 的卷积
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )

        # BN层，处理输出通道数 64 的卷积输出 Tensor
        self.bn1 = nn.BatchNorm2d(64)
        # conv2，它是这个网络结构的第二层，基准输出通道数为 64(之后会扩大)，数量是 layers[0] 个
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        # conv3，它是这个网络结构的第二层，基准输出通道数为 128(之后会扩大)，数量是 layers[1] 个，步长为 2
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1], stride=2)
        # conv4，它是这个网络结构的第二层，基准输出通道数为 256(之后会扩大)，数量是 layers[2] 个，步长为 2
        self.layer3 = self._make_layer(Bottle2neck, 256, layers[2], stride=2)
        # conv5，它是这个网络结构的第二层，基准输出通道数为 512(之后会扩大)，数量是 layers[3] 个，步长为 2
        self.layer4 = self._make_layer(Bottle2neck, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 若步长不为 1 或者 输入通道数不等于输出通道数
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 下采样
            downsample = nn.Sequential(
                # 平均池化，一个 stride x stride 的卷积，步长为 stride，上取整，不计算外围的 0 填充
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                # 一个 1x1 的卷积，步长为 1，不使用偏置，输出通道数为 inplanes，输出通道数为 planes*block.expansion，其中 block 即 Bottleneck
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                # 一个批量归一化层，传入输出通道数
                nn.BatchNorm2d(planes * block.expansion),
            )

        # layers 是一个列表，里面的元素是 Bottleneck。这里是第一个 Bottleneck有 blocks 个 输入通道数为 inplanes，输出通道数为 planes，步长为 stride，
        # 下采样为 downsample，stype为第一个 block 专用的 stage，基准宽度为 baseWidth=26
        layers = [block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth,
                        scale=self.scale)]
        # 输出通道数变为基准输出通道数 planes 乘以 Bottleneck的输出通道倍数
        self.inplanes = planes * block.expansion

        # 然后顺次存入其余的 Bottleneck 到 layers 里
        for i in range(1, blocks):
            # 其余的由于不是新 stage 的第一个 block，所以仅仅给出几个必要的属性，其余的属性按照默认值传入
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        # *layers 表示将列表 layers 分解成 len(layers) 个参数，注入Sequential内，以顺序结构的方式返回当前层
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        # 这部分代码相当于把 conv1, bn1, relu1 的三个操作揉在一起了
        # PS: inplace=True是一个编程参数，常用于函数或方法中。它的基本作用就是在原变量中就地修改内容，而不需要新建一个变量。
        # 这个参数可以使程序更加高效，减少内存占用，同时也可以让程序员更加方便地进行变量操作。
        # 比如，在图像处理中，常常需要对图像进行修改、裁剪、缩放等操作。使用 inplace=true可以直接在原始图像上进行操作，避免新建图像所带来的内存占用。
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # 将 out1 传入一个最大池化层层(一个 3x3 的卷积，步长为 2，填充为 1)处理，就是 layer0(conv1) 之后的max pooling
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        # 把 layer0(conv1) 的输出 out1, 传入 layer1 (这里写法不好，out1应该写 out0 然后 layer1 相当于 layer2 )
        out2 = self.layer1(out1)
        # 把 layer1(conv2) 的输出 out2, 传入 layer2 (吐槽同上)
        out3 = self.layer2(out2)
        # 把 layer2(conv3) 的输出 out3, 传入 layar3 (吐槽同上上)
        out4 = self.layer3(out3)
        # 把 layer3(conv4) 的输出 out4, 传入 layer4 (吐槽同上上上)
        out5 = self.layer4(out4)
        # 返回 conv2 到 conv5 每一层的输出
        return out2, out3, out4, out5

    # 初始化
    def initialize(self):
        # load_state_dict(): 需传入一个字典对象，而非对象的含路径保存文件(self.snapshot)，亦即需要先反序列化字典对象，然后调用该方法。
        # torch.load(): 采用 pickle 将反序列化的对象从存储中加载进来，即把pth文件的内容转成 dict 类型的数据。(序列化将dict变成文本形式，反序列化将文本形式转成dict形式)
        self.load_state_dict(torch.load(self.snapshot), strict=False)


def Res2Net50():
    ''' Res2Net=>__init__(self, layers, snapshot, baseWidth=26, scale=4):
    Args:
        layers => [3, 4, 6, 3]
        snapshot => ../res/res2net50_v1b_26w_4s-3cf99910.pth
    '''
    # 利用 Res2Net 的初始化方法构造了一个 Res2Net，传入了各个层的残差块数量以及预训练模型
    return Res2Net([3, 4, 6, 3], './pretrained_args/res2net50_v1b_26w_4s-3cf99910.pth')


def weight_init(module):  # 初始化权重
    # 遍历 module的所有直接子模块，这里的 n=name, m=module
    for n, m in module.named_children():
        print('initialize: ' + n)
        # 若 module 是 Conv2d 的一个实例
        if isinstance(m, nn.Conv2d):
            # 使用何凯明初始化权重方法，即使用正态分布对权重张量赋值，保留前向传播时权重方差的量级，使用 RELU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # 若 module的 bias 不为 None
            if m.bias is not None:
                # 则用 0 填充 module 的偏置张量
                nn.init.zeros_(m.bias)
        # 若 module 是 BatchNorm2d 或者 InstanceNorm2d 的一个实例
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # 若 module 的 weight 不为空
            if m.weight is not None:
                # 用 1 填充 module 的权重张量
                nn.init.ones_(m.weight)
            # 若 module 的 bias 不为空
            if m.bias is not None:
                # 则用 0 填充 module 的偏置张量
                nn.init.zeros_(m.bias)
        # 若 module 是 Linear 的一个实例
        elif isinstance(m, nn.Linear):
            # 使用何凯明初始化权重方法，即使用正态分布对权重张量赋值，保留前向传播时权重方差的量级，使用 RELU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # 若 module 的 bias 不为空
            if m.bias is not None:
                # 则用 0 填充 module 的偏置张量
                nn.init.zeros_(m.bias)
        # 若 module 是 Sequential 的一个实例
        elif isinstance(m, nn.Sequential):
            # 调用 weight_init 方法，传入 module
            weight_init(m)
        # 若 module 是 RELU 或者 PReLU 的一个实例，忽略
        elif isinstance(m, (nn.ReLU, nn.PReLU)):
            pass
        # 以上都不是的话，调用 initialize 方法
        else:
            m.initialize()
