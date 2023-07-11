@[TOC](第2章 感知机)
# 感知机是什么
**感知机**（perceptron）：算法，本章指“**人工神经元**”或“**朴素感知机**”
感知机有多个输入，一个输出，像电流一样向前方流动（1/传递信号/一个分类得到的结果/“神经元被激活”）或不流动（0/不传递信号/另一个分类相对的结果/神经元被抑制），其中作为的流动过程中的通道的权重类似于电阻，**决定电流流动的难易度**。

**下图是一个接收两个输入信号的感知机的例子**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517182014966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

* x_1、x_2 是输入信号
* y 是输出信号。
* w_1、w_2 是权重。
* ○称为“神经元”或者“节点”。
* 神经元计算出总和`y=w_1*x_1+w_2*x_2`
```javascript
if y<= θ
y=0
else
y=1
```
* 这里将这个界限值称为阈值，用符号 `θ` 表示。
1. 感知机的多个输入信号都有各自固有的权重，**这些权重发挥着控制各个信号的重要性的作用**。
2. 与电阻电流的比例关系相反，权重越大，对应该权重的信号的重要性就越高。
# 简单逻辑电路
与门、或门、与非门都只需确定权重和阀值，参数都是0和1，感知机构造是相同的。
## 与门
当`x_1`、`x_2`作为与门的输入，`w_1`、`w_2`作为与门的权重，`y`作为与门的输出，可以明显得知它的真值表，只有在`x_1`、`x_2`都为1时，`y`才等于1。
根据真值表的条件可以得到很多权重和阀值的结果集，如`(w_1,w_2,θ )=(0.5,0.5,0.7)`、 `(w_1,w_2,θ) =(0.5, 0.5, 0.8)` 、`(w_1,w_2,θ)= (1.0, 1.0, 1.0)` 
## 与非门和或门
由真值表的条件可以得到与非门可以有如下结果集：`(w_1,w_2,θ )=(-0.5,-0.5,-0.7)` 
* 只要把实现与门的参数值的符号取反，就可以实现与非门。

由真值表的条件可以得到或门可以有如下结果集：`(w_1,w_2,θ )=(1,1,0.7)` 

可以知道，我们可以通过人工的方法把有能力赋值的结果集得出来，这个阶段就是我们人在“训练数据”，看哪个数据适用于该输入输出的公式。而机器学习就是把决定参数这个工作交给机器去做。

* 机器需要做的：得到==训练数据==，==学习==来确定参数，
* 人需要做的：建立感知机算法==模型==，整理足够==训练数据==。
# 感知机的实现
## python简单实现
要求：
* 实现与门
* 函数内初始化参数 w1、w2、theta
* 当输入的加权总和超过阈值时返回 1，否则返回 0
```javascript
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
print(AND(1,1))
print(AND(1,0))
```
输出结果：1，0
## 导入权重和偏置
下面我们表示`θ`换为`-b`，得到一个条件式
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517191117186.png)
* 此处，b 称为偏置，w_1 和 w_2 称为权重。
在python解释器中使用Numpy实现上式：
```javascript
>>> import numpy as np
>>> x=np.array([0,1])
>>> w=np.array([0.5,0.5])
>>> b=-0.7
>>> x*w
array([0. , 0.5])
>>> np.sum(x*w)+b
-0.19999999999999996
```
## 使用权重和偏置的实现
与门实现如下：
```javascript
import numpy as np
def AND(x1,x2):
    w=np.array([0.5,0.5])
    x=np.array([x1,x2])
    b=-0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
print(AND(1,0))
```

* 权重是控制输入信号的重要性的参数
* 偏置是调整神经元被激活的容易程度（当输出信号为 1 的程度）的参数

比如，若 b 为 -0.1，则只要输入信号的加权总和超过 0.1，神经元就会被激活。但是如果 b 为 -20.0，则输入信号的加权总和必须超过 20.0，神经元才会被激活。

只需变化上面代码中权重和偏置来实现与非门和或门如下：
```javascript
import numpy as np
def NAND(x1,x2):	#与非门实现
    w=np.array([-0.5,-0.5])
    x=np.array([x1,x2])
    b=0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
print(AND(1,0))
def OR(x1,x2):	#或门实现
    w=np.array([0.5,0.5])
    x=np.array([x1,x2])
    b=-0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
print(OR(1,0))
```
# 感知机的局限性
## 异或门
* 感知机无法实现异或门

1. 或门可以表示如下：

  `(b,w_1,w_2)=(-0.5,1.0,1.0)`情况下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517195313447.png)
以直线-0.5+x1+x2=0分成的两个空间，得到下图（以x1作为横轴，x2作为纵轴）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517195525449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
可以明显看到，○ 表示 0，△ 表示 1，灰色部分为0，白色部分为1

2. 异或门
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517195737821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
一条直线无论如何都不能将上图分为两部分
## 线性和非线性

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517200238964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)		曲线是可以分开的
* 感知机的局限性：只能表示由==一条直线==分割的空间
* 曲线分割而成的空间称为非线性空间
* 直线分割而成的空间称为线性空间
# 多层感知机
**感知机可以通过叠加层来表示异或门**
## 已有门电路的组合
可知异或门的表达式为:

x1$\oplus$x2 = x1$\overline{x2}$+$\overline{x1}$x2
=x1$\overline{x1}$+x1$\overline{x2}$+x2$\overline{x2}$+$\overline{x1}$x2
=x1$\overline{x1x2}$+x2$\overline{x1x2}$
=(x1+x2)$\overline{x1x2}$

通过该式即可用与门或门与非门来表示异或门
即：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517203923657.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 异或门的实现
* 使用之前定义的 AND 函数、NAND 函数、OR 函数实现
```javascript
def XOR(x1,x2):
    s1=OR(x1,x2)
    s2=NAND(x1,x2)
    y=AND(s1,s2)
    return y
print(XOR(1,0))
```
感知机的神经元表示方法表示如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517204603641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 异或门是一种多层结构的神经网络。这里，将最左边的一列称为第 0 层，中间的一列称为第 1 层，最右边的一列称为第 2 层
* 与门、或门是单层感知机，而异或门是 2 层感知机(这里是按照权重的层数来算的)
* 叠加了多层的感知机也称为多层感知机(multi-layered perceptron）
* 这种 2 层感知机的运行过程可以比作流水线的组装作业：
	
	**第 1 段（第 1 层）的工人对传送过来的零件进行加工，完成后再传送给第 2 段（第 2 层）的工人。第 2 层的工人对第 1 层的工人传过来的零件进行加工，完成这个零件后出货（输出）。**

# 从与非门到计算机
**多层感知机可以实现比之前见到的电路更复杂的电路**
如：
* 加法运算的加法器
* 二进制转换为十进制的编码器、
* 满足某些条件就输出 1 的电路
* 计算机：
**计算机和感知机一样，也有输入和输出，会按照某个既定的规则进行计算，通过与非门的组合，就能再现计算机进行的处理**

《计算机系统要素：从零开始构建现代计算机》这本书以深入理解计算机为主题，论述了通过 NAND构建可运行俄罗斯方块的计算机的过程。

*2 层感知机（严格地说是激活函数使用了非线性的 sigmoid 函数的感知机，具体请参照下一章）可以表示任意函数。但是，使用 2 层感知机的构造，通过设定合适的权重来构建计算机是一件非常累人的事情。实际上，在用与非门等低层的元件构建计算机的情况下，分阶段地制作所需的零件（模块）会比较自然，即先实现与门和或门，然后实现半加器和全加器，接着实现算数逻辑单元（ALU），然后实现 CPU。*

最后我们可以知道用叠加感知机可以表示非线性函数。

end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
本文章大量引用原书中的内容和训练营中课程内容
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^1]的读书笔记

[^1]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)


