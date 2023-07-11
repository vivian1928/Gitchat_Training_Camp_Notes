@[TOC](第4章 神经网络的学习)
* 学习的目的在于寻找损失函数最小的权重参数
# 4.1　从数据中学习
* 感知机收敛定理可以在有限次学习中使感知机解决线性可分问题
## 4.1.1　数据驱动
1. 如何识别`5`
* 从图像中提取特征量（将输入图像提取最重要数据的转换器，来转换为向量形式，如：`SIFT`、`SURF`、`HOG`），再通过分类器（`SVM`、`KNN`等）学习。
* 机器学习中，人工设置图像特征量
* 深度学习（端到端机器学习）中，特征量都是机器学习的，即输入量直接可以作为原始数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190520173708478.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 4.1.2　训练数据和测试数据
* 训练数据也称为`监督数据`（监督着学习过程的数据）
* 为了正确评价模型的泛化能力，所以要划分数据（必须要有大量没有见过的数据）
* `泛化能力`指处理没见过的数据的能力
* 只对一个数据集过度拟合的问题叫做`过拟合`
# 4.2　损失函数
损失函数表示性能在多大程度上不拟合，即==恶劣程度==
用损失函数的==负数==表示性能在多大程度上拟合，即==优良程度==
* 损失函数可以使用任意函数，一般用`均方误差`和`交叉熵误差`等。
## 4.2.1　均方误差
公式如下：
$$E=\frac{1}{2}\sum_{k}(y_k-t_k)^2$$
* $y_k$表示神经网络的输出
* $t_k$表示监督数据
* $k$表示数据维度

python实现如下：
```javascript
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)
```
例子如下：
```javascript
import numpy as np
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

t=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y=np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(mean_squared_error(y,t))

y=np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(mean_squared_error(y,t))
```
1. y是softmax函数的输出，所以表示概率。
2. t是已知正误的监督数据，用one-hot来表示数据集，标签表示2是正确的。
3. 此处神经网络输出概率最大的是2

运行结果如下：
0.09750000000000003
0.5975
可以知道，对于神经网络模型得到输出概率最高与监督数据不符时，损失函数的结果会增大，表示恶劣程度较高
## 4.2.2　交叉熵误差
公式如下：
$$E=-\sum_{k}t_k\ln{y_k}$$
* 只计算正确解标签所对应的自然函数的负数

$-\ln{x}$对应的函数图像如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190520191121378.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
而$$0\leq{yk}\leq1$$
所以输出的概率越小损失函数越大，和监督数据越不吻合。
python实现如下：
```javascript
def cross_entropy_error(y,t):
    delta=1e-7 #delta用来防止log(0)=infinity的出现
    return -np.sum(t*np.log(y+delta)) 
```
同样按照均方误差的例子来调用，得到的运行结果如下：
0.510825457099338
2.302584092994546
## 4.2.3　mini-batch 学习
之前我们将一个数据条的损失函数讨论了，如果有N个数据，每个数据有分别有很多需要对比的元素，则交叉熵误差公式如下：
$$E=-\frac{1}{N}\sum_{N}\sum_{k}t_{nk}\ln{y_{nk}}$$
* $t_{nk}$是第N个数据的第k个元素的监督数据
* $y_{nk}$是第N个数据的第k个元素的输出
* 除以 N 进行正规化，还可以求每个数据的平均损失函数

如果数据量特别大时，求和会是一件很耗时的工作，所以这时候只用一部份，作为全部数据的“近似”，称为`小批量（mini-batch）`，如用60000个数据取100个学习，称为 mini-batch 学习。

这时我们回忆上一章读入mnist数据集的情况：
* 读入MNIST 数据后，训练数据有 `60000` 个，输入数据是 `784` 维（28 × 28）的图像数据，监督数据是 `10` 维的数据。因此，上面的 `x_train、t_train` 的形状分别是 `(60000, 784)` 和 `(60000, 10)`。

在其中随机抽取10个数据用`np.random.choice(train_size, batch_size)`代码：
```javascript
train_size = x_train.shape[0] #训练数据第0维数值，代表有多少个数据
batch_size = 10 #要选出多少数据
batch_mask = np.random.choice(train_size, batch_size) #选出的数据位置放入batch_mask
x_batch = x_train[batch_mask] #把这些位置上的输入数据取出来放入x_batch作为mini—batch学习的输入数据
t_batch = t_train[batch_mask] #把这些位置上的监督数据取出来放入t_batch作为mini—batch学习的监督数据
```
## 4.2.4　mini-batch 版交叉熵误差的实现
1. one-hot格式的监督数据
```javascript
def cross_entropy_error(y, t):
    if y.ndim == 1: #一维数组要把第一维放到第二维，表示只有一条数据
        t = t.reshape(1, t.size) #reshape函数代两维的参数
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] #记下第一维的数值，表示有多少条数据
    return -np.sum(t * np.log(y + 1e-7)) / batch_size 
```
2. 非one-hot格式的监督数据
```javascript
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))/ batch_size #根据start与stop指定的范围以及step设定的步长，生成一个 array,一个参数batch_size时，从0到batch_size，step=1，生成一个数组。
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190520235027664.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 4.2.5　为何要设定损失函数
识别精度：相当于监督数据是n个t，k个输出y，识别出了m个，识别精度是m/n
损失函数：相当于监督数据有n个t，k个输出y，识别出了m个，用一个函数function（n，k）拟合出两个之间的差别程度，损失函数越小，说明识别的越符合
* 识别精度对于参数微小变化反馈不大，可能100个数据，调整某个参数，还是能识别32个数据，描述函数变化的导数为0，无法相应去知道怎么调整参数
* 而对于损失函数：*如果导数的值为负，通过使该权重参数向正方向改变，可以减小损失函数的值；反过来，如果导数的值为正，则通过使该权重参数向负方向改变，可以减小损失函数的值。不过，当导数的值为 0 时，无论权重参数向哪个方向变化，损失函数的值都不会改变，此时该权重参数的更新会停在此处。*

end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^1]的课程读书笔记
* 本文章大量引用原书中的内容和训练营课程中的内容作为笔记

[^1]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)
