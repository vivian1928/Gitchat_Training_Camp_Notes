@[TOC](第7章 卷积神经网络)
# 7.3　池化层
* 池化运算：缩小高、长方向上的空间的运算
* 一般来说，池化的窗口大小会和步幅设定成相同的值
* 除了Max池化，还有Average池化，在图像识别领域，主要使用 Max 池化

按步幅 2 进行 2 × 2 的 Max 池化的例子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527134953786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
池化层的特征：
1. 没有要学习的参数，没有滤波器（权重）的卷积运算过程，只是把相应的值通过制定的条件取出来。
2. 通道数不发生变化，通道之间==独立==进行运算。
3. 对微小的位置变化具有鲁棒性（健壮），数据在池化窗口里的一些微小变化，可能并不会很大程度上影响输出数据。
# 7.4　卷积层和池化层的实现
## 7.4.1　4 维数组
* 我们知道在CNN个层之间传递的是4维数据，如 (10, 1, 28, 28)，表示10 个高为 28、长为 28、通道为 1 的数据

用python随机生成：
```javascript
>>> x = np.random.rand(10, 1, 28, 28) # 随机生成数据
>>> x.shape
(10, 1, 28, 28)
```
访问第n个数据：
```javascript
>>> x[0].shape # (1, 28, 28)
>>> x[1].shape # (1, 28, 28)
>>> x[n-1].shape # (1, 28, 28)
```
访问第 n 个数据的第 n 个通道的==空间==数据（二维）:
```javascript
>>> x[n-1, n-1] # 或者x[n-1][n-1]
```
## 卷积运算的技巧：基于 im2col 的展开
我们知道4维数组各个位置的计算要嵌套4层for循环，为了避免这种复杂和耗时长的方式，所以我们应用im2col函数来进行数组的展开。

im2col函数做卷积运算的大致过程：
1.  输入数据，将应用到滤波器的区域（3 维方块）==横向==展开为 1 列
2. 再将卷积层的滤波器（权重）==纵向==展开为 1 列
3. 做矩阵乘法
4. 批处理集合起来，重新转换为合适的4维数组形状

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527141300555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 7.4.3　卷积层的实现
im2col 函数接口：

    im2col (input_data, filter_h, filter_w, stride=1, pad=0)

`input_data`——由（数据量，通道，高，长）的 4 维数组构成的输入数据
`filter_h`——滤波器的高
`filter_w`——滤波器的长
`stride`——步幅
`pad`——填充

im2col函数的具体应用：
```javascript
import sys, os
sys.path.append(os.pardir)
from common.util import im2col  #从封装的common.util里引入im2col函数

x1 = np.random.rand(1, 3, 7, 7) #随机生成一个1个数据，3个通道，长宽为7*7的数据集
col1 = im2col(x1, 5, 5, stride=1, pad=0) #滤波器大小为5*5，步幅1，不填充
print(col1.shape) # 调用后得到(9, 75)形状的2维数组

x2 = np.random.rand(10, 3, 7, 7) #随机生成一个10个数据，3个通道，、长宽为7*7的数据集
col2 = im2col(x2, 5, 5, stride=1, pad=0) #滤波器大小为5*5，步幅1，不填充
print(col2.shape) # 调用后得到(90, 75)形状的2维数组
```
第一维数据`90`代表：每一个通道用滤波器产生的数据条的个数为`3*3=9`
第二位数据`75`代表：滤波器的展开大小，`3`通道`（5*5）`的滤波器输出为`75`
`10`个数据累加到每用一次滤波器产生的数据大小上
过程图如下[^1]：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527182608329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
卷积层的实现：
```javascript
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W. #W二维大小的滤波器
        self.b = b  #b二维大小的偏置
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)  #计算出来输出的二维大小
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  #输入数据的横向展开
        col_W = self.W.reshape(FN, -1).T # 滤波器的展开，FN行，-1模糊控制列数，(10, 3, 5, 5) 形状的数组，就会转换成 (10, 75) 形状的数组，再将数组行列转置成为（75，10）纵向展开
        out = np.dot(col, col_W) + self.b #进行矩阵乘法求和

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  #模糊C的大小，等到N、out_h、out_w算出来后生成，并且用transpose改变索引位置

        return out
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527183955485.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 7.4.4　池化层的实现
与卷积层不同的点：
* 池化的应用区域按通道单独展开

对输入数据展开池化的应用区域（2×2 的池化的例子）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052718413317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
展开时：
第一维数据`12`代表：通道数乘以每个通道可以展开的数据条数
第二维数据`4`代表：池化窗口里的数据个数，即池化窗口高度乘以池化窗口宽度

池化层实现流程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527185138159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
im2col函数做池化运算的大致过程：
1.  输入数据展开
2. 取最大值成为1列
3. 根据输出大小重新转换为合适的数组形状

池化层的实现：
```javascript
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride) 
	```
	池化层输出大小计算公式：out_h=1+(H-pool_h)/stirde，H为输入高度，pool_h为池化窗口高度，stride为步幅
	```
        out_w = int(1 + (W - self.pool_w) / self.stride) 
	```
	池化层输出大小计算公式：out_w=1+(W-pool_w)/stirde，W为输入宽度，pool_w为池化窗口宽度，stride为步幅
	```
        # 展开(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w) #行数由计算后决定

        # 最大值(2)
        out = np.max(col, axis=1) 按照第一维求最大值，即跨列求出每一行的最大值
        # 转换(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
```

end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^2]的课程读书笔记
* 本文章大量引用原书中的内容和训练营课程中的内容作为笔记

[^1]:[ im2col的原理和实现：作者dwyane12138](https://blog.csdn.net/dwyane12138/article/details/78449898)
[^2]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)

