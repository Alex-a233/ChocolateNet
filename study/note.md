####请问人类肠道里的息肉总共有多少种？

人类肠道里的息肉主要分为以下几种类型:

1. 腺瘤性息肉(Adenomatous polyps):这是最常见的一种息肉,约占总息肉的70-80%。它们通常是增生性的,有可能发展成为恶性肿瘤(结直肠癌)。
2. 过性息肉(Hyperplastic polyps):约占总息肉的15-20%,这种息肉通常无恶性潜能。
3. 炎症性息肉(Inflammatory polyps):通常由长期炎症引起,约占5-10%。
4. 过性增生息肉(Serrated polyps):包括锯齿状腺瘤、锯齿状息肉和微小锯齿状息肉,约占5%左右。这种息肉也可能发展为恶性肿瘤。
5. 混合型息肉:包含两种或两种以上的息肉类型。

总的来说,肠道息肉的主要类型大致有4-5种,但具体数量会根据患者的病理情况而有所不同。定期进行肠道检查很重要,可以及时发现和治疗各种类型的肠道息肉。

####按照形态划分呢？

从形态上来看,肠道息肉主要可以分为以下几种类型:

1. 有蒂息肉(Pedunculated polyps)：这种息肉有一个细长的茎部,息肉头部悬垂在肠道内。约占总息肉的20-30%。
2. 无蒂息肉(Sessile polyps)：这种息肉直接长在肠壁上,没有明显的茎部。约占总息肉的60-70%。
3. 平坦息肉(Flat polyps)：这种息肉表面平坦,很难从周围肠道粘膜上识别出来。约占总息肉的10%左右。
4. 微小息肉(Diminutive polyps)：小于5mm的息肉,约占总息肉的30-50%。
5. 大息肉(Large polyps)：直径超过1cm的息肉,约占总息肉的10-20%。

这些不同形态的息肉在临床诊断和治疗策略上都有一定的差异。临床医生通常会根据息肉的形态、大小等特征来制定合适的治疗方案。

####请分别详细讲述这5种息肉的特征及其危险性？

好的,让我分别详细介绍一下这5种肠道息肉的特点和危险性:

1. 有蒂息肉(Pedunculated polyps):

   - 特点:这种息肉长在一条细长的茎柄上,悬垂在肠腔内。
   - 危险性:由于茎部较细,这种息肉容易扭曲、出血或被肠蠕动推离原位,因此有较高的出血和肠梗阻等并发症风险。但对于发生癌变的风险,并不高于其他类型的息肉。
2. 无蒂息肉(Sessile polyps):

   - 特点:这种息肉直接生长在肠壁上,没有明显的茎部。
   - 危险性:无蒂息肉与结直肠癌的发生风险较有蒂息肉更高。这是因为它们更容易发生腺瘤性变化,从而进展为恶性肿瘤。
3. 平坦息肉(Flat polyps):

   - 特点:这种息肉表面平坦,与周围粘膜难以区分。
   - 危险性:平坦息肉往往更难发现和切除,而且更容易发生腺瘤性变化和癌变。因此,它们的恶性转化风险较高。
4. 微小息肉(Diminutive polyps):

   - 特点:直径小于5mm的小型息肉。
   - 危险性:虽然个体风险较低,但由于数量众多,微小息肉仍可能累积发生癌变。
5. 大息肉(Large polyps):

   - 特点:直径超过1cm的大型息肉。
   - 危险性:大息肉更容易发生出血、梗阻等并发症,且恶性转化的风险也较高。因此需要更加重视和及时治疗。

总之,肠道息肉的形态和大小会影响其临床表现和恶性转化的风险。临床医生需要根据这些特征综合评估患者,制定合理的诊治方案。定期进行肠镜筛查十分重要,能够及时发现和切除各种类型的息肉。

#### 请问 挤压激励 和 通道注意力 有何区别？

挤压激励和通道注意力之间有以下几个主要区别:

1. 操作对象不同:

   - 挤压激励是对整个特征图进行操作,通过全局平均池化和全连接层来获得通道级的特征权重。
   - 通道注意力是对通道维度进行建模和加权,提高有用信息的权重。
2. 计算复杂度不同:

   - 挤压激励相对较为简单,只需要全局池化和两个全连接层。
   - 通道注意力需要使用卷积或者全连接层来建模通道之间的依赖关系,计算量较大。
3. 关注点不同:

   - 挤压激励更加关注整体特征的重要性,忽略了通道之间的关联。
   - 通道注意力则重点关注不同通道特征之间的互相影响和依赖。
4. 应用场景不同:

   - 挤压激励可以应用于不同任务和网络结构中,提升整体特征的表达能力。
   - 通道注意力更适用于需要建模通道间关系的复杂网络结构,如图像分类、目标检测等。

总的来说,挤压激励和通道注意力都是常见的注意力机制,侧重点不同,适用场景也有所区别,需要根据具体任务和模型特点来选择合适的方法。

#### 我做语义分割的话，这两种方法选哪个比较好？

对于语义分割任务来说,通道注意力机制相比于挤压激励通常更加适合。

原因如下:

1. 语义分割需要对不同类别物体进行精准的像素级分割,这需要网络能够很好地捕捉不同通道特征之间的依赖关系。
2. 通道注意力可以自适应地增强那些对分割任务更加重要的通道特征,提高网络的区分能力。
3. 相比于全局性的挤压激励,通道注意力可以更好地建模局部空间信息,从而得到更精细的分割结果。
4. 许多最新的语义分割网络如PSPNet、DeepLabV3+都采用了通道注意力机制,取得了不错的效果。

当然,具体选择哪种注意力机制还需要结合具体的网络架构和数据集特点来权衡。但总的来说,对于语义分割这种需要细粒度特征表达的任务,通道注意力机制更加合适和有效。

#### 语义分割中的 mIoU 和 mDice 有何区别？

mIoU 和 mDice 都是常见的语义分割任务的评价指标,两者都用于评估模型在预测像素级别分类标签时的整体表现。不过它们有一些区别:

1. 计算方式不同:

   - mIoU (mean Intersection over Union)是计算预测结果与真实标签之间的交并比的平均值。
   - mDice (mean Dice Coefficient)是计算预测结果与真实标签的Dice系数的平均值。
2. 对错误的敏感程度不同:

   - mIoU更倾向于惩罚大的预测错误,对大的偏差比较敏感。
   - mDice对小的预测错误更加敏感,更多地关注整体的相似程度。
3. 数值范围不同:

   - mIoU的取值范围是0到1,0表示完全不重合,1表示完全重合。
   - mDice的取值范围也是0到1,但0表示完全不重合,1表示完全重合。

总的来说,mIoU和mDice都是衡量语义分割模型预测性能的常用指标,根据实际需求可以选用其中一种或两种进行评估。
mIoU更注重大错误,而mDice则更关注整体的相似程度。在实际应用中,通常需要结合多种指标来综合评估模型的性能。

#### 请问 PyTorch 的 torch.nn.functional.interpolate 方法使用 size 和 scale_factor 采样有何区别？速度上有何区别？

`torch.nn.functional.interpolate` 方法的 `size` 和 `scale_factor` 参数确实有所不同:

1. **size**:

   - 使用 `size` 参数时,需要提供明确的输出尺寸,例如 `(H, W)` 或 `(D, H, W)`。
   - 根据提供的目标尺寸,插值算法会被用来调整输入的尺寸。
   - 这种方式可以更精确地控制输出的尺寸。
2. **scale_factor**:

   - 使用 `scale_factor` 参数时,需要提供一个缩放因子,例如 `2.0` 或 `(2.0, 2.0)` 或 `(1.0, 2.0, 2.0)`。
   - 根据给定的缩放因子,插值算法会被用来调整输入的尺寸。
   - 这种方式不会直接指定目标尺寸,而是根据缩放因子来计算。

总的来说:

- 使用 `size` 可以更精确地控制输出尺寸,但需要明确指定目标尺寸。
- 使用 `scale_factor` 可以更方便地对输入进行缩放,但不能直接控制输出尺寸。

具体使用哪种方式,取决于你的应用场景和需求。

关于 `torch.nn.functional.interpolate` 方法中使用 `size` 和 `scale_factor` 两种采样方式的执行速度差异,主要有以下几点:

1. **计算复杂度**:

   - 使用 `size` 参数时,需要根据给定的目标尺寸计算插值系数,这涉及更多的计算。
   - 而使用 `scale_factor` 时,只需要根据缩放因子计算,计算复杂度相对更低。
2. **内存访问**:

   - 使用 `size` 参数时,需要根据目标尺寸进行内存访问和索引,可能会导致更多的内存访问开销。
   - 使用 `scale_factor` 时,由于只需要进行缩放,内存访问模式相对更简单,开销也会更小。
3. **框架优化**:

   - PyTorch 框架对这两种方式的优化程度可能会有所不同。
   - 对于某些特定的硬件和场景,框架可能会针对性地优化 `scale_factor` 的实现,从而提升执行速度。

总的来说,**使用 `scale_factor` 通常可以获得更好的执行速度**,因为它的计算复杂度和内存访问开销相对更低。但这种差异也会受到具体硬件、输入规模以及框架优化程度的影响。

在实际应用中,如果对执行速度有严格要求,可以尝试两种方式并进行测试,选择最优的方式。同时也要考虑其他因素,如灵活性和精确性需求等。

####请问训练的比较好的分割网络的GPU使用率是怎么变化的？

分割网络的GPU使用率会随着训练过程而发生变化,这主要取决于以下几个因素:

1. 网络结构复杂度:

   - 更复杂的网络结构(如更多层、更多参数)通常会占用更多的GPU资源,导致GPU使用率较高。
2. 输入图像尺寸:

   - 图像尺寸越大,需要的GPU内存和计算量越大,GPU使用率也会相应提高。
3. 批量大小(Batch Size):

   - 批量大小越大,GPU利用率通常会更高,但同时也需要更多的显存。
4. 特征提取和分割两个阶段:

   - 特征提取阶段通常会占用较多的GPU资源,而分割阶段相对轻量一些。
5. 训练过程中的优化:

   - 在训练初期,模型参数更新较快,GPU使用率较高。随着训练进行,模型收敛,GPU使用率会逐渐降低。

总的来说,分割网络的GPU使用率会随着网络复杂度、输入尺寸、批量大小以及训练进度等因素的变化而动态变化。可以通过监控GPU使用情况来优化训练配置,提高训练效率。

#### 语义分割模型一般训练多少个周期比较合适？

语义分割模型的训练周期长度主要取决于以下几个因素:

1. 数据集大小和复杂度:

   - 数据集较大且包含复杂场景时,通常需要更长的训练周期才能收敛。
2. 模型复杂度:

   - 更复杂的模型通常需要更长的训练周期来学习所有的参数。简单的模型可能只需要几十个epoch就可以收敛。
3. 超参数设置:

   - 学习率、批量大小等超参数的选择会影响模型的收敛速度。通常需要调整这些参数以达到最佳效果。
4. 任务难度:

   - 一些复杂的语义分割任务,如城市场景、医疗图像等,可能需要更长的训练周期。

一般来说,对于大多数语义分割任务,训练 50-200 个 epoch 是比较合适的范围。具体情况需要根据上述因素进行实验和调整。一些经验做法包括:

1. 先以较少的 epoch 训练,观察模型在验证集上的表现。
2. 如果模型还没有收敛,可以增加训练 epoch 数直到验证集性能稳定。
3. 同时监控训练过程中的loss曲线变化,如果出现过拟合,可以适当减少 epoch 数或调整其他超参数。

总之,语义分割模型的训练周期长度需要根据具体任务和数据集进行实验,并通过观察模型在验证集上的表现来确定最佳的训练轮数。

#### x1 = conv(x) 经过卷积处理之后的原特征图x也会发生变化吗？

是的,在经过 `conv(x)` 之后,原始的特征图 `x` 也会发生变化。具体来说:

1. 卷积操作本身就会改变特征图的尺寸和通道数:

   - 假设输入特征图 `x` 的尺寸为 `(C_in, H, W)`,经过卷积层后,输出特征图的尺寸会变为 `(C_out, H_out, W_out)`,其中 `C_out` 是输出通道数,`H_out` 和 `W_out` 取决于卷积核的大小、步长以及填充方式等。
2. 卷积操作会改变特征图的数值分布:

   - 卷积操作相当于对输入特征图进行加权线性组合,这会改变特征图中每个元素的数值。
   - 比如原始特征图 `x` 中某个位置的值可能是 `10`,经过卷积后变成了 `5.3`。
3. 卷积后的特征图可能具有不同的语义意义:

   - 卷积层能够提取输入特征图中的局部特征,比如边缘、纹理等,这些特征对于后续的特征学习和模式识别很重要。
   - 因此,卷积后的特征图不仅数值发生变化,而且也包含了不同层次的语义信息。

总之,`x1 = conv(x)` 这一操作会从多个方面改变原始的特征图 `x`,包括尺寸、数值分布以及语义含义等。这些变化为后续的特征提取和模型训练奠定了基础。

#### 能给我讲讲内镜医师区分肠道 息肉边界 的方法吗？

内镜医师判断肠道息肉的边界主要有以下几种方法:

1. 观察息肉表面特点:

- 息肉边界通常表现为粘膜突然变化,从正常粘膜到息肉表面有明显界限。
- 通常息肉表面光滑较为光洁,而正常粘膜表面有一定的皱褶纹理。

2. 观察血管分布:

- 正常粘膜下有环绕的血管网,而息肉部位血管分布异常,常见无规则分布或血管稀疏。

3. 活检取样检查:

- 在疑似息肉边界处取活检样本,通过病理学检查确定息肉的界限。

4. 放大内镜观察:

- 利用高倍放大功能,可以更清晰地观察息肉与正常粘膜的界线特征。

5. 染色辅助识别:

- 使用亚甲蓝等染色剂涂抹,可以增强息肉与正常粘膜的对比,有利于识别边界。

综合运用这些方法,结合临床经验,内镜医师能够较为准确地判断息肉的边界,为后续诊治提供依据。

#### debug时如何查看一个tensor里的所有数据

右键点击需要查看的tensor然后点evaluate expression选项(Alt+F8)，输入 tensor.cpu().numpy() 或者 np.array(tensor.data.cpu())

#### 深度学习中的normalization和regularization究竟应该如何翻译？

深度学习中的 normalization 和 regularization 这两个概念在中文中确实没有一个统一的翻译,但常见的说法如下:

1. Normalization (归一化)

   - 这个概念通常被翻译为"归一化"或"规范化"。它指的是对数据或中间层的激活值进行缩放和平移,使其服从某种分布,如均值为 0、方差为 1 的标准正态分布。
   - 常见的归一化方法有批量归一化(Batch Normalization)、层归一化(Layer Normalization)等。
2. Regularization (正则化)

   - 这个概念通常被翻译为"正则化"或"规则化"。它指的是在损失函数中加入额外的项,以防止模型过拟合,提高其泛化能力。
   - 常见的正则化方法有 L1 正则化、L2 正则化、Dropout 等。

总的来说:

- Normalization 关注的是特征或激活值的分布,而 Regularization 关注的是模型整体的泛化性能。
- Normalization 主要应用在模型的中间层,而 Regularization 主要应用在整个模型的损失函数上。
- 两者都是深度学习中常用的技术,在提高模型性能方面起着重要作用。

因此,建议在中文表述时,可以分别使用"归一化"和"正则化"这两个术语。当需要更加精确地表述时,也可以使用"规范化"和"规则化"作为替代。

#### sa/ca的4种情况

##### case 1
t * sa(x) + t & t * ca(x) + t

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python .\train.py
Training start and it is time about 2024-08-14 19:35:59.507567
current time 2024-08-14 19:36:20.618345, epoch [001/060], step [0010/0091], loss:1.105005]
current time 2024-08-14 19:36:25.124100, epoch [001/060], step [0020/0091], loss:0.865719]
current time 2024-08-14 19:36:29.814206, epoch [001/060], step [0030/0091], loss:0.478080]
current time 2024-08-14 19:36:34.413779, epoch [001/060], step [0040/0091], loss:0.537183]
current time 2024-08-14 19:36:39.038882, epoch [001/060], step [0050/0091], loss:0.298564]
current time 2024-08-14 19:36:43.715012, epoch [001/060], step [0060/0091], loss:0.322902]
current time 2024-08-14 19:36:48.330155, epoch [001/060], step [0070/0091], loss:0.393134]
current time 2024-08-14 19:36:53.019298, epoch [001/060], step [0080/0091], loss:0.369544]
current time 2024-08-14 19:36:57.709206, epoch [001/060], step [0090/0091], loss:0.306635]
current time 2024-08-14 19:36:58.178253, epoch [001/060], step [0091/0091], loss:0.560732]
current time 2024-08-14 19:37:26.810532, epoch 1 & best model's mdice 0.571743859649123
dataset: CVC-300 & mdice: 0.627000
dataset: CVC-ClinicDB & mdice: 0.683723
dataset: CVC-ColonDB & mdice: 0.560646
dataset: ETIS-LaribPolypDB & mdice: 0.435681
dataset: Kvasir & mdice: 0.778019
current time 2024-08-14 19:37:38.616508, epoch [002/060], step [0010/0091], loss:0.415245]
current time 2024-08-14 19:37:43.292961, epoch [002/060], step [0020/0091], loss:0.251475]
current time 2024-08-14 19:37:47.935899, epoch [002/060], step [0030/0091], loss:0.196316]
current time 2024-08-14 19:37:52.689646, epoch [002/060], step [0040/0091], loss:0.391812]
current time 2024-08-14 19:37:57.475311, epoch [002/060], step [0050/0091], loss:0.214339]
current time 2024-08-14 19:38:02.056530, epoch [002/060], step [0060/0091], loss:0.240422]
current time 2024-08-14 19:38:06.571671, epoch [002/060], step [0070/0091], loss:0.259248]
current time 2024-08-14 19:38:11.109595, epoch [002/060], step [0080/0091], loss:0.204140]
current time 2024-08-14 19:38:15.676276, epoch [002/060], step [0090/0091], loss:0.192206]
current time 2024-08-14 19:38:16.020028, epoch [002/060], step [0091/0091], loss:0.197719]
current time 2024-08-14 19:38:44.979839, epoch 2 & best model's mdice 0.621898120300752
dataset: CVC-300 & mdice: 0.689562
dataset: CVC-ClinicDB & mdice: 0.726032
dataset: CVC-ColonDB & mdice: 0.605634
dataset: ETIS-LaribPolypDB & mdice: 0.505795
dataset: Kvasir & mdice: 0.806103

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python .\train.py
Training start and it is time about 2024-08-14 20:00:39.691647
current time 2024-08-14 20:00:56.781015, epoch [001/060], step [0010/0091], loss:1.748868]
current time 2024-08-14 20:01:01.345301, epoch [001/060], step [0020/0091], loss:0.956753]
current time 2024-08-14 20:01:05.909420, epoch [001/060], step [0030/0091], loss:0.855990]
current time 2024-08-14 20:01:10.412483, epoch [001/060], step [0040/0091], loss:0.879858]
current time 2024-08-14 20:01:14.978800, epoch [001/060], step [0050/0091], loss:0.509154]
current time 2024-08-14 20:01:19.513865, epoch [001/060], step [0060/0091], loss:0.578256]
current time 2024-08-14 20:01:24.064992, epoch [001/060], step [0070/0091], loss:0.506592]
current time 2024-08-14 20:01:28.615644, epoch [001/060], step [0080/0091], loss:0.479716]
current time 2024-08-14 20:01:33.418401, epoch [001/060], step [0090/0091], loss:0.409645]
current time 2024-08-14 20:01:33.838536, epoch [001/060], step [0091/0091], loss:0.458251]
current time 2024-08-14 20:02:03.981744, epoch 1 & best model's mdice 0.4351238095238097
dataset: CVC-300 & mdice: 0.378788
dataset: CVC-ClinicDB & mdice: 0.544492
dataset: CVC-ColonDB & mdice: 0.421031
dataset: ETIS-LaribPolypDB & mdice: 0.321008
dataset: Kvasir & mdice: 0.678337
current time 2024-08-14 20:02:14.853362, epoch [002/060], step [0010/0091], loss:0.296382]
current time 2024-08-14 20:02:19.447269, epoch [002/060], step [0020/0091], loss:0.247924]
current time 2024-08-14 20:02:23.980784, epoch [002/060], step [0030/0091], loss:0.229834]
current time 2024-08-14 20:02:28.540036, epoch [002/060], step [0040/0091], loss:0.307743]
current time 2024-08-14 20:02:33.200494, epoch [002/060], step [0050/0091], loss:0.324711]
current time 2024-08-14 20:02:37.774886, epoch [002/060], step [0060/0091], loss:0.230363]
current time 2024-08-14 20:02:42.450656, epoch [002/060], step [0070/0091], loss:0.225221]
current time 2024-08-14 20:02:47.298282, epoch [002/060], step [0080/0091], loss:0.211462]
current time 2024-08-14 20:02:52.242147, epoch [002/060], step [0090/0091], loss:0.284495]
current time 2024-08-14 20:02:52.632768, epoch [002/060], step [0091/0091], loss:0.268746]
current time 2024-08-14 20:03:20.977576, epoch 2 & best model's mdice 0.5659542606516291
dataset: CVC-300 & mdice: 0.584975
dataset: CVC-ClinicDB & mdice: 0.683116
dataset: CVC-ColonDB & mdice: 0.546582
dataset: ETIS-LaribPolypDB & mdice: 0.443660
dataset: Kvasir & mdice: 0.795212

##### case 2
t * sa(x) & t * ca(x)

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python .\train.py
Training start and it is time about 2024-08-14 19:56:26.766645
current time 2024-08-14 19:56:43.630278, epoch [001/060], step [0010/0091], loss:1.036459]
current time 2024-08-14 19:56:48.274583, epoch [001/060], step [0020/0091], loss:0.710142]
current time 2024-08-14 19:56:53.042925, epoch [001/060], step [0030/0091], loss:0.533130]
current time 2024-08-14 19:56:57.778734, epoch [001/060], step [0040/0091], loss:0.327218]
current time 2024-08-14 19:57:02.470212, epoch [001/060], step [0050/0091], loss:0.364748]
current time 2024-08-14 19:57:07.192751, epoch [001/060], step [0060/0091], loss:0.346039]
current time 2024-08-14 19:57:11.883915, epoch [001/060], step [0070/0091], loss:0.445174]
current time 2024-08-14 19:57:16.606527, epoch [001/060], step [0080/0091], loss:0.296955]
current time 2024-08-14 19:57:21.296486, epoch [001/060], step [0090/0091], loss:0.320694]
current time 2024-08-14 19:57:21.734472, epoch [001/060], step [0091/0091], loss:0.328121]
current time 2024-08-14 19:57:52.047920, epoch 1 & best model's mdice 0.596416290726817
dataset: CVC-300 & mdice: 0.652955
dataset: CVC-ClinicDB & mdice: 0.703005
dataset: CVC-ColonDB & mdice: 0.577270
dataset: ETIS-LaribPolypDB & mdice: 0.482834
dataset: Kvasir & mdice: 0.791787
current time 2024-08-14 19:58:03.163958, epoch [002/060], step [0010/0091], loss:0.490662]
current time 2024-08-14 19:58:07.778171, epoch [002/060], step [0020/0091], loss:0.244759]
current time 2024-08-14 19:58:12.389781, epoch [002/060], step [0030/0091], loss:0.156917]
current time 2024-08-14 19:58:16.924982, epoch [002/060], step [0040/0091], loss:0.349010]
current time 2024-08-14 19:58:21.443761, epoch [002/060], step [0050/0091], loss:0.137393]
current time 2024-08-14 19:58:25.994696, epoch [002/060], step [0060/0091], loss:0.187769]
current time 2024-08-14 19:58:30.563104, epoch [002/060], step [0070/0091], loss:0.296345]
current time 2024-08-14 19:58:35.305133, epoch [002/060], step [0080/0091], loss:0.238658]
current time 2024-08-14 19:58:40.170586, epoch [002/060], step [0090/0091], loss:0.220238]
current time 2024-08-14 19:58:40.562170, epoch [002/060], step [0091/0091], loss:0.291321]
current time 2024-08-14 19:59:08.947689, epoch 2 & best model's mdice 0.6379403508771926
dataset: CVC-300 & mdice: 0.720820
dataset: CVC-ClinicDB & mdice: 0.771579
dataset: CVC-ColonDB & mdice: 0.620355
dataset: ETIS-LaribPolypDB & mdice: 0.514856
dataset: Kvasir & mdice: 0.813425

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python train.py
Training start and it is time about 2024-08-14 20:11:37.597673
current time 2024-08-14 20:11:54.226056, epoch [001/060], step [0010/0091], loss:1.136797]
current time 2024-08-14 20:11:59.164643, epoch [001/060], step [0020/0091], loss:0.658231]
current time 2024-08-14 20:12:03.727808, epoch [001/060], step [0030/0091], loss:0.529862]
current time 2024-08-14 20:12:08.338524, epoch [001/060], step [0040/0091], loss:0.503278]
current time 2024-08-14 20:12:12.980048, epoch [001/060], step [0050/0091], loss:0.378932]
current time 2024-08-14 20:12:17.574862, epoch [001/060], step [0060/0091], loss:0.570981]
current time 2024-08-14 20:12:22.122623, epoch [001/060], step [0070/0091], loss:0.323538]
current time 2024-08-14 20:12:26.702166, epoch [001/060], step [0080/0091], loss:0.340624]
current time 2024-08-14 20:12:31.234429, epoch [001/060], step [0090/0091], loss:0.474602]
current time 2024-08-14 20:12:31.625062, epoch [001/060], step [0091/0091], loss:0.450175]
current time 2024-08-14 20:12:58.087211, epoch 1 & best model's mdice 0.5496372180451127
dataset: CVC-300 & mdice: 0.559955
dataset: CVC-ClinicDB & mdice: 0.674098
dataset: CVC-ColonDB & mdice: 0.537696
dataset: ETIS-LaribPolypDB & mdice: 0.412229
dataset: Kvasir & mdice: 0.780977
current time 2024-08-14 20:13:08.798939, epoch [002/060], step [0010/0091], loss:0.218200]
current time 2024-08-14 20:13:13.630137, epoch [002/060], step [0020/0091], loss:0.183955]
current time 2024-08-14 20:13:18.475622, epoch [002/060], step [0030/0091], loss:0.171760]
current time 2024-08-14 20:13:23.180763, epoch [002/060], step [0040/0091], loss:0.241456]
current time 2024-08-14 20:13:27.840359, epoch [002/060], step [0050/0091], loss:0.237110]
current time 2024-08-14 20:13:32.609207, epoch [002/060], step [0060/0091], loss:0.348848]
current time 2024-08-14 20:13:37.181229, epoch [002/060], step [0070/0091], loss:0.215064]
current time 2024-08-14 20:13:41.723555, epoch [002/060], step [0080/0091], loss:0.256402]
current time 2024-08-14 20:13:46.264464, epoch [002/060], step [0090/0091], loss:0.196698]
current time 2024-08-14 20:13:46.608582, epoch [002/060], step [0091/0091], loss:0.192516]
current time 2024-08-14 20:14:14.800235, epoch 2 & best model's mdice 0.6053708020050123
dataset: CVC-300 & mdice: 0.638175
dataset: CVC-ClinicDB & mdice: 0.742553
dataset: CVC-ColonDB & mdice: 0.595007
dataset: ETIS-LaribPolypDB & mdice: 0.463123
dataset: Kvasir & mdice: 0.818824

##### case 3
t + sa(x) & t + ca(x)

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python train.py
Training start and it is time about 2024-08-14 20:23:28.580146
current time 2024-08-14 20:23:45.074478, epoch [001/060], step [0010/0091], loss:1.055376]
current time 2024-08-14 20:23:49.732868, epoch [001/060], step [0020/0091], loss:0.975379]
current time 2024-08-14 20:23:54.379216, epoch [001/060], step [0030/0091], loss:0.543417]
current time 2024-08-14 20:23:58.967560, epoch [001/060], step [0040/0091], loss:0.667515]
current time 2024-08-14 20:24:03.578917, epoch [001/060], step [0050/0091], loss:0.470577]
current time 2024-08-14 20:24:08.161749, epoch [001/060], step [0060/0091], loss:0.290832]
current time 2024-08-14 20:24:12.697217, epoch [001/060], step [0070/0091], loss:0.340640]
current time 2024-08-14 20:24:17.201922, epoch [001/060], step [0080/0091], loss:0.360224]
current time 2024-08-14 20:24:21.751363, epoch [001/060], step [0090/0091], loss:0.474772]
current time 2024-08-14 20:24:22.157622, epoch [001/060], step [0091/0091], loss:0.280107]
current time 2024-08-14 20:24:48.880353, epoch 1 & best model's mdice 0.48433734335839607
dataset: CVC-300 & mdice: 0.484822
dataset: CVC-ClinicDB & mdice: 0.581010
dataset: CVC-ColonDB & mdice: 0.465502
dataset: ETIS-LaribPolypDB & mdice: 0.368445
dataset: Kvasir & mdice: 0.722832
current time 2024-08-14 20:24:59.447580, epoch [002/060], step [0010/0091], loss:0.235446]
current time 2024-08-14 20:25:04.002617, epoch [002/060], step [0020/0091], loss:0.350379]
current time 2024-08-14 20:25:08.516670, epoch [002/060], step [0030/0091], loss:0.236363]
current time 2024-08-14 20:25:13.079602, epoch [002/060], step [0040/0091], loss:0.192399]
current time 2024-08-14 20:25:17.586325, epoch [002/060], step [0050/0091], loss:0.262067]
current time 2024-08-14 20:25:22.113236, epoch [002/060], step [0060/0091], loss:0.347151]
current time 2024-08-14 20:25:26.623984, epoch [002/060], step [0070/0091], loss:0.209513]
current time 2024-08-14 20:25:31.348363, epoch [002/060], step [0080/0091], loss:0.229881]
current time 2024-08-14 20:25:36.097453, epoch [002/060], step [0090/0091], loss:0.233731]
current time 2024-08-14 20:25:36.474244, epoch [002/060], step [0091/0091], loss:0.197759]
current time 2024-08-14 20:26:06.273773, epoch 2 & best model's mdice 0.6026141604010024
dataset: CVC-300 & mdice: 0.657858
dataset: CVC-ClinicDB & mdice: 0.723581
dataset: CVC-ColonDB & mdice: 0.587570
dataset: ETIS-LaribPolypDB & mdice: 0.468213
dataset: Kvasir & mdice: 0.815063

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python train.py
Training start and it is time about 2024-08-14 20:36:36.267429
current time 2024-08-14 20:36:52.153474, epoch [001/060], step [0010/0091], loss:0.990692]
current time 2024-08-14 20:36:56.895267, epoch [001/060], step [0020/0091], loss:0.762369]
current time 2024-08-14 20:37:01.558734, epoch [001/060], step [0030/0091], loss:0.559178]
current time 2024-08-14 20:37:06.403996, epoch [001/060], step [0040/0091], loss:0.509168]
current time 2024-08-14 20:37:11.062513, epoch [001/060], step [0050/0091], loss:0.527293]
current time 2024-08-14 20:37:15.845773, epoch [001/060], step [0060/0091], loss:0.353753]
current time 2024-08-14 20:37:20.503043, epoch [001/060], step [0070/0091], loss:0.306688]
current time 2024-08-14 20:37:25.256942, epoch [001/060], step [0080/0091], loss:0.322853]
current time 2024-08-14 20:37:30.164555, epoch [001/060], step [0090/0091], loss:0.215681]
current time 2024-08-14 20:37:30.603034, epoch [001/060], step [0091/0091], loss:0.305421]
current time 2024-08-14 20:38:01.069242, epoch 1 & best model's mdice 0.5670966165413532
dataset: CVC-300 & mdice: 0.631657
dataset: CVC-ClinicDB & mdice: 0.672140
dataset: CVC-ColonDB & mdice: 0.554862
dataset: ETIS-LaribPolypDB & mdice: 0.433003
dataset: Kvasir & mdice: 0.772548
current time 2024-08-14 20:38:11.899422, epoch [002/060], step [0010/0091], loss:0.337265]
current time 2024-08-14 20:38:16.404892, epoch [002/060], step [0020/0091], loss:0.338213]
current time 2024-08-14 20:38:21.003954, epoch [002/060], step [0030/0091], loss:0.276791]
current time 2024-08-14 20:38:25.570255, epoch [002/060], step [0040/0091], loss:0.461264]
current time 2024-08-14 20:38:30.212139, epoch [002/060], step [0050/0091], loss:0.382952]
current time 2024-08-14 20:38:34.859280, epoch [002/060], step [0060/0091], loss:0.238794]
current time 2024-08-14 20:38:39.486795, epoch [002/060], step [0070/0091], loss:0.364118]
current time 2024-08-14 20:38:44.129722, epoch [002/060], step [0080/0091], loss:0.327127]
current time 2024-08-14 20:38:48.709507, epoch [002/060], step [0090/0091], loss:0.168766]
current time 2024-08-14 20:38:49.068911, epoch [002/060], step [0091/0091], loss:0.148009]
current time 2024-08-14 20:39:16.558148, epoch 2 & best model's mdice 0.6290412280701754
dataset: CVC-300 & mdice: 0.699937
dataset: CVC-ClinicDB & mdice: 0.742700
dataset: CVC-ColonDB & mdice: 0.611489
dataset: ETIS-LaribPolypDB & mdice: 0.511839
dataset: Kvasir & mdice: 0.812449

##### case 4
sa(x) & ca(x)

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python train.py
Training start and it is time about 2024-08-14 20:42:35.362847
current time 2024-08-14 20:42:51.109462, epoch [001/060], step [0010/0091], loss:0.829962]
current time 2024-08-14 20:42:55.742968, epoch [001/060], step [0020/0091], loss:0.511130]
current time 2024-08-14 20:43:00.394360, epoch [001/060], step [0030/0091], loss:0.581253]
current time 2024-08-14 20:43:04.895625, epoch [001/060], step [0040/0091], loss:0.445253]
current time 2024-08-14 20:43:09.428171, epoch [001/060], step [0050/0091], loss:0.388017]
current time 2024-08-14 20:43:13.944781, epoch [001/060], step [0060/0091], loss:0.278351]
current time 2024-08-14 20:43:18.476872, epoch [001/060], step [0070/0091], loss:0.299163]
current time 2024-08-14 20:43:22.994379, epoch [001/060], step [0080/0091], loss:0.349612]
current time 2024-08-14 20:43:27.495363, epoch [001/060], step [0090/0091], loss:0.424051]
current time 2024-08-14 20:43:27.885990, epoch [001/060], step [0091/0091], loss:0.364566]
current time 2024-08-14 20:43:58.127833, epoch 1 & best model's mdice 0.5219976190476191
dataset: CVC-300 & mdice: 0.522022
dataset: CVC-ClinicDB & mdice: 0.639669
dataset: CVC-ColonDB & mdice: 0.512275
dataset: ETIS-LaribPolypDB & mdice: 0.383796
dataset: Kvasir & mdice: 0.756847
current time 2024-08-14 20:44:08.789323, epoch [002/060], step [0010/0091], loss:0.277664]
current time 2024-08-14 20:44:13.385644, epoch [002/060], step [0020/0091], loss:0.215530]
current time 2024-08-14 20:44:17.952789, epoch [002/060], step [0030/0091], loss:0.208028]
current time 2024-08-14 20:44:22.541219, epoch [002/060], step [0040/0091], loss:0.163194]
current time 2024-08-14 20:44:27.107612, epoch [002/060], step [0050/0091], loss:0.211754]
current time 2024-08-14 20:44:31.714680, epoch [002/060], step [0060/0091], loss:0.338157]
current time 2024-08-14 20:44:36.293932, epoch [002/060], step [0070/0091], loss:0.296657]
current time 2024-08-14 20:44:40.889343, epoch [002/060], step [0080/0091], loss:0.197675]
current time 2024-08-14 20:44:45.468443, epoch [002/060], step [0090/0091], loss:0.180621]
current time 2024-08-14 20:44:45.812207, epoch [002/060], step [0091/0091], loss:0.212945]
current time 2024-08-14 20:45:14.039427, epoch 2 & best model's mdice 0.61564649122807
dataset: CVC-300 & mdice: 0.642767
dataset: CVC-ClinicDB & mdice: 0.751098
dataset: CVC-ColonDB & mdice: 0.599197
dataset: ETIS-LaribPolypDB & mdice: 0.489205
dataset: Kvasir & mdice: 0.825729

(chocolateNet) PS D:\Study\pyspace\ChocolateNet> python train.py
Training start and it is time about 2024-08-14 20:49:35.789260 
current time 2024-08-14 20:49:51.260822, epoch [001/060], step [0010/0091], loss:0.927190]
current time 2024-08-14 20:49:55.887321, epoch [001/060], step [0020/0091], loss:0.658128]
current time 2024-08-14 20:50:00.393899, epoch [001/060], step [0030/0091], loss:0.579867]
current time 2024-08-14 20:50:04.927149, epoch [001/060], step [0040/0091], loss:0.453068]
current time 2024-08-14 20:50:09.554007, epoch [001/060], step [0050/0091], loss:0.603060]
current time 2024-08-14 20:50:14.087121, epoch [001/060], step [0060/0091], loss:0.466277]
current time 2024-08-14 20:50:18.620322, epoch [001/060], step [0070/0091], loss:0.238807]
current time 2024-08-14 20:50:23.200456, epoch [001/060], step [0080/0091], loss:0.290529]
current time 2024-08-14 20:50:27.733019, epoch [001/060], step [0090/0091], loss:0.375269]
current time 2024-08-14 20:50:28.139318, epoch [001/060], step [0091/0091], loss:0.304487]
current time 2024-08-14 20:50:58.048555, epoch 1 & best model's mdice 0.5695850877192984
dataset: CVC-300 & mdice: 0.598002
dataset: CVC-ClinicDB & mdice: 0.670719
dataset: CVC-ColonDB & mdice: 0.560612
dataset: ETIS-LaribPolypDB & mdice: 0.445694
dataset: Kvasir & mdice: 0.766755
current time 2024-08-14 20:51:08.820886, epoch [002/060], step [0010/0091], loss:0.266864]
current time 2024-08-14 20:51:13.382360, epoch [002/060], step [0020/0091], loss:0.266197]
current time 2024-08-14 20:51:17.994813, epoch [002/060], step [0030/0091], loss:0.354762]
current time 2024-08-14 20:51:22.740588, epoch [002/060], step [0040/0091], loss:0.350076]
current time 2024-08-14 20:51:27.537364, epoch [002/060], step [0050/0091], loss:0.139817]
current time 2024-08-14 20:51:32.347628, epoch [002/060], step [0060/0091], loss:0.439116]
current time 2024-08-14 20:51:37.099942, epoch [002/060], step [0070/0091], loss:0.262280]
current time 2024-08-14 20:51:42.003301, epoch [002/060], step [0080/0091], loss:0.296185]
current time 2024-08-14 20:51:46.946354, epoch [002/060], step [0090/0091], loss:0.225682]
current time 2024-08-14 20:51:47.306246, epoch [002/060], step [0091/0091], loss:0.204931]
current time 2024-08-14 20:52:14.731539, epoch 2 & best model's mdice 0.6134968671679197
dataset: CVC-300 & mdice: 0.662092
dataset: CVC-ClinicDB & mdice: 0.747194
dataset: CVC-ColonDB & mdice: 0.601939
dataset: ETIS-LaribPolypDB & mdice: 0.477331
dataset: Kvasir & mdice: 0.812254
