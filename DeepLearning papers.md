### 深度学习论文学习

​		乍一看到某个问题，你会觉得很简单，其实你并没有理解其复杂性。当你把问题搞清楚之后，又会发现真的很复杂，于是你就拿出一套复杂的方案来。实际上，你的工作只做了一半，大多数人也都会到此为止……。但是，真正伟大的人还会继续向前，直至**找到问题的关键和深层次原因**，然后再拿出一个**优雅的**、**堪称完美的**有效方案。

​																									—— from 乔布斯

#### AlexNet

* 核心思想
* 创新点
  * 首次采用ReLU激活函数，极大增大收敛速度且从根本上解决了梯度消失问题
  * 完全采用有监督训练。也正因为如此，DL的主流学习方法也因此变为了纯粹的有监督学习
  * 第一次使用GPU加速

#### ResNet

* https://www.cnblogs.com/shine-lee/p/12363488.html

* 解决的问题：深层神经网络在理论上来说，会比浅层神经网络的表现要好（起码不会比浅层要差），但是实际来看，深层神经网络会出现“**退化**”的现象，即**给网络叠加更多的层后，性能却快速下降的情况**，ResNet的提出，就是从模型结构的角度去解决这一问题

* 主要思想：ResNet从结构入手，基本结构如下图所示，F(x)在训练过程中，可以由网络自动学习，如果**网络学习的F(x)为0**，那么我们的网络具备identity（浅层）的拟合能力（起码不会比浅层差，而是与浅层一样的效果）；如果网络学习的**F(x)为其他函数**，那么模型的表现能力相较原来的identity会得到提升，我们给每个block开出一条支路的目的，是为了让网络具有选择的“权利”，避免深层“退化”（出现退化的话，F(x)可以学习成0）.

  <img src="./images/image-20221021210621846.png" alt="image-20221021210621846" style="zoom:50%;" />

* 网络结构：<img src="./images/image-20221021211408779.png" alt="image-20221021211408779" style="zoom:50%;" />

#### RNN

* https://blog.csdn.net/v_JULY_v/article/details/89894058：由RNN到**LSTM**

* 想让神经网络有“记忆”，上一次的输出先保存起来，需要作为下一次的输入

  <img src="./images/image-20221020115724172.png" alt="image-20221020115724172" style="zoom:50%;" />

* RNN可以被看做是同一神经网络的多次复制，每个神经网络模块会把消息传递给下一个<img src="./images/image-20221020140637185.png" alt="image-20221020140637185" style="zoom:50%;" />

* 双向RNN：看到的范围更广<img src="./images/image-20221020120054289.png" alt="image-20221020120054289" style="zoom:50%;" />

* 局限性： RNN 会受到**短时记忆**的影响。如果一条序列足够长，那它们将**很难将信息从较早的时间步传送到后面的时间步**；原因是在递归神经网络中，RNN的**早期层**获得**小梯度更新会停止学习**。 由于这些层不学习，RNN 可以忘记它在较长序列中看到的内容，因此具有短时记忆

* 由RNN局限性引出LSTM（Long Short-Term Model）模型：

  * 记忆能力有限，记住最重要的，忘记无关紧要的

  * <img src="./images/image-20221020141549518.png" alt="image-20221020141549518" style="zoom:50%;" />

  * **LSTM的“门结构”**

    LSTM有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。**门是一种让信息选择式通过的方法**。他们包含一个**sigmoid**神经网络层和一个**pointwise乘法的非线性操作**。

    如此，0代表“不许任何量通过”，1就指“允许任意量通过”！从而使得网络就能了解哪些数据是需要遗忘，哪些数据是需要保存。

    * 忘记门：			<img src="./images/image-20221020164627502.png" alt="image-20221020164627502" style="zoom: 50%;" />
      * 该忘记门会读取上一个输出![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155706754786217349.svg)和当前输入![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155706756757918470.svg)，做一个Sigmoid 的非线性映射，然后输出一个向量![f_{t}](https://private.codecogs.com/gif.latex?f_%7Bt%7D)（该向量每一个维度的值都在0到1之间，1表示完全保留，0表示完全舍弃，相当于记住了重要的，忘记了无关紧要的），最后与细胞状态![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155706757691155460.svg)相乘
    * 输入门：            <img src="./images/image-20221020165011359.png" alt="image-20221020165011359" style="zoom:50%;" />
      * 确定什么样的新信息被存放在细胞状态中，包含两个部分：
        第一，sigmoid层称“输入门层”决定什么值我们将要更新；
        第二，一个tanh层创建一个新的候选值向量![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155706765888087563.svg)，会被加入到状态中。 
    * **更新细胞状态**： <img src="./images/image-20221020165322890.png" alt="image-20221020165322890" style="zoom:50%;" />
      * 我们把旧状态与![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415570677073428868.svg)相乘，丢弃掉我们确定需要丢弃的信息。接着加上![img](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155706771729335767.svg)。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。
    * 输出门：![image-20221020165731650](./images/image-20221020165731650.png)
      * 首先，我们运行一个sigmoid层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过tanh进行处理（得到一个在-1到1之间的值）并将它和sigmoid门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。

#### VGG（visual geometry group by Oxford university）

* <img src="./images/image-20221018110842592.png" alt="image-20221018110842592" style="zoom:50%;" />
* VGG参数非常多，主要原因是第一层起flatten作用的全连接层的参数过多
* VGG的前两层卷积层占据了绝大部分内存来计算（收到图片输入大小影响）
* VGG**全部使用3×3卷积核**的原因：利用两层3×3卷积核去代替一个5×5卷积核，由于深度变深，模型的非线性能力更强，且参数数量也较少（**核心的思想为使用多层3×3卷积核去代替调更大的一层卷积核来减少参数加深深度**）
* 对比实验，探究深度对网络性能的影响（数据、算力、算法 深度学习三驾马车）
* https://zhuanlan.zhihu.com/p/41423739

#### GAN

* 核心思想
  * Generative Adversarial Networks
  * 损失函数：![image-20221015163802591](./images/image-20221015163802591.png)
  * ![image-20221015164052674](./images/image-20221015164052674.png)
* 创新点
  * 生成两个模型，一个是**生成**模型，一个是辨别模型，两者相对抗，使得生成模型更接近“真实”，最后使辨别器“无能为力”
* 可借鉴的使用方法
  * two sample test ：判断两块数据是不是来自于同一个分布（在数据科学中会使用T分布检测），就训练一个二分类分类器就好
  * KL散度：
* 缺点
  * 不稳定，收敛性不好

* https://distill.pub/2021/gnn-intro/)

#### Attention

* 想解决什么问题

  * 解决输入不只是一个向量，且每个向量长度不一样的问题，解决多输入且输入相关联的问题，self-attention层考虑上下文，然后给出一个输入的输出，再作为之后的网络的输入

    <img src="./images/image-20221016160845090.png" alt="image-20221016160845090" style="zoom: 33%;" />

* 原理：

  * <img src="./images/image-20230220214019968.png" alt="image-20230220214019968" style="zoom: 33%;" />

  * 1.**计算不同向量之间的关联性α**：（有多种方法）

    * 点乘模块来计算相关系数：<img src="./images/image-20221016161245386.png" alt="image-20221016161245386" style="zoom:33%;" />
    * 将各个输入向量乘以相应权重参数，得到新的向量后，**用这些新的向量来计算向量之间的相关性**，而**引入权重参数的目的是让网络能够进行学习**，不断地调整权重参数来更好的分析各个输入向量之间的相关性（用softmax是经验问题）

    <img src="./images/image-20221016161448226.png" alt="image-20221016161448226" style="zoom:33%;" />

  * 2.根据每个向量之间的关联性来抽取重要的信息：

    * 将各个原始的输入再乘以另一个权重参数得到新向量V，V与第一个输入向量对应的相关系数相乘再全部相加，来**得到针对于第一个输入向量的输出值b1**，**这个值是同时考虑了其他输入向量后得到的一个向量，包含了更多的信息**

    <img src="./images/image-20221016161927598.png" alt="image-20221016161927598" style="zoom:33%;" />

  * 1、2两步的并行矩阵计算方法：

    （1）<img src="./images/image-20221016162350453.png" alt="image-20221016162350453" style="zoom:33%;" />

    （2）![image-20221016162622496](./images/image-20221016162622496.png)

    （3）<img src="./images/image-20221016162806424.png" alt="image-20221016162806424" style="zoom: 50%;" />

    （4）<img src="./images/image-20221016162917412.png" alt="image-20221016162917412" style="zoom:33%;" />

    **引入了q、k、v三个可网络学习的参数去学习输入向量之间的重要性和关联度**

    （5）拓展：**多头机制**，**引入更多参数**进行学习,不同的参数负责不同的相关性，多头机制丰富了这种相关性，具体做法就是在基础的参数（**q、k、v**）上，再衍生出多个新的参数（**qi1，qi2;ki1,ki2;vi1,vi2**），然后各自学习各自的相关性（qi1与ki1作运算）<img src="./images/image-20221016163145578.png" alt="image-20221016163145578" style="zoom:50%;" />

* 在自注意力机制中，学习不到输入向量之间的位置关系（时序、顺序关系），若输入数据的位置关系对预测结果有影响的话，需要考虑位置关系，这时就需要引入能够表征位置关系的编码：

  ​							   <img src="./images/image-20221016163555144.png" alt="image-20221016163555144" style="zoom: 50%;" />

* **将Attention应用在图片处理上：**

  * <img src="./images/image-20230306204010508.png" alt="image-20230306204010508" style="zoom:50%;" />

  * 把每个像素看成一个三维的向量，整张图片就有（5×10）个三维向量，这样就可以使用Attention来处理图片

  * **Attention与CNN的联系**：CNN可以说是简化版的自注意力机制

    * Attention在对一个像素进行计算时，是考虑了图片的其他所有向量，然后进行学习，最后自己学出来一个形状的窗口，而CNN只考虑一个固定窗口范围内的向量

    ![image-20221016164155414](./images/image-20221016164155414.png)

  * **Attention与RNN的关系**：Attention为并行计算，在计算速度上比RNN更快<img src="./images/image-20221016164835488.png" alt="image-20221016164835488" style="zoom:50%;" />


#### Transformer

​	<img src="./images/image-20221017171400812.png" alt="image-20221017171400812" style="zoom:50%;" />

* Sequence to sequence Model

  * 输入向量的长度与输出向量的长度没有绝对的联系，输出向量的长度由模型自己决定
    * 常用于NLP中QA的任务

    * 也可以用于多标签的分类任务

    * 可以用于目标检测

  * 普遍的架构：
    * <img src="./images/image-20230307174810718.png" alt="image-20230307174810718" style="zoom:33%;" />

* Beam Search

* **Encoder and Decoder**

  * **Cross attention**：

  <img src="./images/image-20221017170945553.png" alt="image-20221017170945553" style="zoom: 33%;" />

  

  * Encoder：

    * <img src="./images/image-20221017165025227.png" alt="image-20221017165025227" style="zoom: 33%;" />

    * <img src="./images/image-20230307190547325.png" alt="image-20230307190547325" style="zoom:33%;" />

    * 对于第一个block：**首先做一个self-attention，并做残差连接，最后做一个归一化，得到第一个输出，随后将他作为全连接神经网络的输入（依旧做残差连接），再做一次归一化得到最后的输出**；上述整个过程为一个block，即为transformer中的encoder架构<img src="./images/image-20230307185909411.png" alt="image-20230307185909411" style="zoom: 50%;" />

  * Decoder：

    * Decoder 是在 Encoder的基础上，增**加了一个接收Encoder输出的block，进行Cross attention计算** ，然后**最初的输入是一个特殊的Masked Self-attention层**(考虑了时序关系、前后关系，**只看当前位置前方的信息**)

    <img src="./images/image-20230307193702882.png" alt="image-20230307193702882" style="zoom: 50%;" />

    * **Cross attention**：
      * ![image-20230307193959826](./images/image-20230307193959826.png)
    * **encoder与decoder联合运作的过程**：<img src="./images/image-20230307191657388.png" alt="image-20230307191657388" style="zoom: 33%;" />
      * <img src="./images/image-20221017165801776.png" alt="image-20221017165801776" style="zoom: 33%;" />
    * NAT的表现一般没有AT好<img src="./images/image-20221017170612262.png" alt="image-20221017170612262" style="zoom:50%;" />

#### BERT（Bidirectional Encoder Representation from Transformers）

* 

#### VIT

* 

#### MAE

* 

#### Contrast learning(Moco/SimCLR)

* 

#### CLIP（Contrastive Language-Image Pre-training）

![image-20230219200438288](./images/image-20230219200438288.png)

#### DELLE2（OpenAI的图片生成模型）

* 将CLIP模型和GLIDE模型融合到了一起![image-20230113145154429](./images/image-20230113145154429.png)
* <img src="./images/image-20230117164822727.png" alt="image-20230117164822727" style="zoom:50%;" /><img src="./images/image-20230117164923983.png" alt="image-20230117164923983" style="zoom:50%;" />

#### DMTet

[Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis 论文总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/609866304?utm_campaign=shareopn&utm_medium=social&utm_oi=938846415775129600&utm_psn=1616450170987433984&utm_source=wechat_session)

* 



#### GET3D

[GET3D 论文解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/568878981?utm_campaign=shareopn&utm_medium=social&utm_oi=938846415775129600&utm_psn=1616448856408932352&utm_source=wechat_session)

* 主要贡献：可以生成具有贴图的三维网格模型，可以直接用于下游任务
* 相关工作对比：
  * 体素模型、点云模型等的生成工作，不能将其直接应用于下游任务（非网格模型）
  * 大部分工作生成的模型没有纹理
* 具体方法：
  * 几何生成部分：关键模型：DMTet
  * 纹理生成部分

#### Embedding

* https://blog.csdn.net/baimafujinji/article/details/77836142

#### GNN（Graph Neural Networks）
