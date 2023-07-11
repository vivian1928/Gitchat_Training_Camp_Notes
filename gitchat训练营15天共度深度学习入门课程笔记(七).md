@[TOC](第5章 误差反向传播法)
# 5.1　计算图
* 高效计算权重参数的方法：误差反向传播法
* 误差反向传播法的两种表示方法：数学式和计算图
## 5.1.1　用计算图求解
*问题 1：太郎在超市买了 2 个 100 日元一个的苹果，消费税是 10%，请计算支付金额。*
○中表示计算内容：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522162423200.png)
○中表示计算方式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522162833594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
问题 2：太郎在超市买了 2 个苹果、3 个橘子。其中，苹果每个 100 日元，橘子每个 150 日元。消费税是 10%，请计算支付金额。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522162856708.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
计算图解题流程：从左到右==正向传播==
## 5.1.2　局部计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522163017840.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
把上面的复杂计算当作一个模块，这个模块的实现过程不用了解，只需要得到它的结果，用它的结果做加和运算。
## 5.1.3　为何用计算图解题
1. 利用看起来简单的局部计算达到简化计算的目的。
2. 可以保存中间结果
3. 可以更方便地进行反向传播的计算

对于问题一，当我们想知道增加苹果价格会对最终的支付金额有什么==影响==时，我们就要求支付金额对于苹果价格的导数，设苹果的价格为 x，支付金额为 L，则相当于求$$ \frac{\partial L}{\partial x}$$。
即反向传播求导数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522163701961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
可以看到，支付金额关于苹果价格的导数中间带入了苹果和消费税的乘积，得到的结果为 2.2 日元，这说明如果苹果的价格上涨 1 日元，最终的支付金额会增加 2.2 日元
# 5.2　链式法则
## 5.2.1　计算图的反向传播
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522164433571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
反向传播就是下层传来的信号$E$反向通过函数计算$f$时，乘这个函数对上层向下传播的信号的偏导$\frac{\partial{y}}{\partial{x}}$
即：*计算图的反向传播：沿着与正方向相反的方向，乘上局部导数*
例如：假设 $y=f(x)=x^2$，则局部导数为 $\frac{\partial y}{\partial x}=2x$
## 5.2.2　什么是链式法则
例子：
对于复合函数$z=(x+y)^2$，可以分成以下两个式子
$$z=t^2$$
$$t=x+y$$
==链式法则==：*如果某个函数由复合函数表示，则该复合函数的导数可以用构成复合函数的各个函数的导数的乘积表示。*
对于上式来说：
$$\frac{\partial{z}}{\partial{x}}=\frac{\partial{z}}{\partial{t}}*\frac{\partial{t}}{\partial{x}}$$
$$\frac{\partial z}{\partial t}=2t$$ 
$$\frac{\partial t}{\partial x}=1$$
$$\frac{\partial{z}}{\partial{x}}=2t=2(x+y)$$
对于问题一来说：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522171215976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 5.2.3　链式法则和计算图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522171520789.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 最开始处的输入$\frac{\partial{z}}{\partial{z}}$=1省略了
* 在计算图上，链式法则反向传播，每一条链上下层输入乘以变量的偏导
# 5.3　反向传播
## 5.3.1　加法节点的反向传播

对于$$z = x + y$$
$$\frac{\partial{z}}{\partial{x}}=1$$
$$\frac{\partial{z}}{\partial{y}}=1$$

    加法运算节点反向传播乘1

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522171831273.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
例子如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522171928833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 5.3.2　乘法节点的反向传播
对于$$z = xy$$
$$\frac{\partial{z}}{\partial{x}}=y$$
$$\frac{\partial{z}}{\partial{y}}=x$$

    乘法运算节点反向传播乘翻转值
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522172535384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
例子如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052217255569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## 5.3.3　苹果的例子
那我们回到最初的问题一，`苹果的价格`、`苹果的个数`、`消费税`这 3 个变量各自如何影响最终支付的金额。![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052217380468.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
可以得到下图括号中的内容如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190522174332187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^1]的课程读书笔记
* 本文章大量引用原书中的内容和训练营课程中的内容作为笔记

[^1]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)
