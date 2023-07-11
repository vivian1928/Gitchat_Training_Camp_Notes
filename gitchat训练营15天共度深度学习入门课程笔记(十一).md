@[TOC](第7章 卷积神经网络)

基于全连接层（Affine 层）的网络的例子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526114302505.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
基于 CNN 的网络的例子，新增了 Convolution 层和 Pooling 层（用灰色的方块表示）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526114342302.png)
CNN常见结构：
* “Convolution - ReLU -（Pooling）”（Pooling 层有时会被省略）
* 靠近输出的层中使用“Affine - ReLU”
* 最后的输出层中使用“Affine - Softmax”

# 7.2　卷积层
* 特征图（feature map）：卷积层的输入输出数据
* 输入特征图（input feature map）：卷积层的输入数据
* 输出特征图（output feature map）：卷积层的输出数据
## 7.2.2　卷积运算
 * 卷积运算相当于图像处理中的“滤波器运算”
 * 用“$\circledast$”符号表示卷积运算
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526115144596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

卷积计算的过程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526115336425.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

上面的卷积运算以1的步幅在输入数据上向右向下滑动滤波器的窗口，每滑动一次就将各个位置上滤波器的元素和输入的对应元素相乘，然后再求和（有时将这个计算称为乘积累加运算）。然后，将这个结果保存到输出的对应位置。

偏置存在的情况：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526115653387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 7.2.3　填充
* 填充（padding）：在进行卷积层的处理之前，有时要向输入数据的周围填入固定的数据（比如 0 等）（==为了调整输出的大小，防止输出越来越小，变为1==）
* 幅度为 1 的填充：指用幅度为 1 像素的 0 填充周围
* 增大填充后，输出大小会变大

例子如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526120520799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 7.2.4　步幅
* 步幅（stride）：应用滤波器的位置间隔
* 增大步幅后，输出大小会变小

![在这里插入图片描述](https://img-blog.csdnimg.cn/201905261210120.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 输出数据大小计算
* 输入大小为 `(H, W)`
* 滤波器大小为` (FH, FW)`
* 输出大小为` (OH, OW)`
* 填充为` P`，步幅为 `S`

$$OH=\frac{H+2P-FH}{S}+1$$
$$OW=\frac{W+2P-FW}{S}+1$$
上式要可以除尽，否则需要报错，在一些深度学习框架中会四舍五入，不需要报错。
## 7.2.5　3 维数据的卷积运算
当有了纵深（通道方向）时，卷积计算的例子：

卷积计算：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526121823723.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 按通道进行输入数据和滤波器的卷积运算，并将结果相加
* 输入数据和滤波器的通道数要设为相同的值

## 7.2.6　结合方块思考
结合长方体来思考卷积运算：
1. 通道数为 1 的输出数据:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526122305227.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
3. 通道数为FN的输出数据：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526122323150.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

应用多个滤波器就可以得到多个卷积计算的输出。

3. 有偏置的情况：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526122436259.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

CNN的处理流：用多个滤波器得到了FN个输出数据，即（FH，OH，OW）的数据方块，将该方块传递给下一层继续处理。
## 7.2.7　批处理
* 卷积运算批处理：各层间的数据按 (batch_num, channel, height, width) 的顺序保存为 4 维数据

一次对 N 个数据进行卷积运算的例子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190526123440659.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)


end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^1]的课程读书笔记
* 本文章大量引用原书中的内容和训练营课程中的内容作为笔记

[^1]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)
