@[TOC](第5章 误差反向传播法)
# 5.4　简单层的实现
## 5.4.1　乘法层的实现
```javascript
class MulLayer: #乘法层（类）
    def __init__(self):	#初始化变量x，y，保存上层传来的两个输入
        self.x = None 
        self.y = None

    def forward(self, x, y): #正向传播输出
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout): #反向传播两个分支输出
        dx = dout * self.y # 上游传来的导数结果乘以翻转量
        dy = dout * self.x 

        return dx, dy
```
代码实例化买两个苹果问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523133421920.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 正向传播：

```javascript
apple = 100 #苹果的单价
apple_num = 2 #苹果的数目
tax = 1.1 #价格税

# layer
mul_apple_layer = MulLayer() #实例化计算苹果总价层
mul_tax_layer = MulLayer() #实例化计算完价格税的总价层

# forward
apple_price = mul_apple_layer.forward(apple, apple_num) #向前传播得出结果
price = mul_tax_layer.forward(apple_price, tax)

print(price) # 220
```
* 反向传播：
 
```javascript
dprice = 1 #正向传播最后的输出对自己的偏导为1
dapple_price, dtax = mul_tax_layer.backward(dprice) #反向传播求出前一层苹果总价的导数和价格税的导数
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #苹果总价的导数再向前传播求出苹果单价的导数和苹果数目的导数

print(dapple, dapple_num, dtax) # 2.2 110 200
```
## 5.4.2　加法层的实现
```javascript
class AddLayer: #加法层（类）
    def __init__(self): #初始化类，由于加法层两个分支和输入是一样的，直接把输入的导数原封不动的流向输出，所以处理过程不需要记住上层传来的输入
        pass

    def forward(self, x, y): #正向传播
        out = x + y
        return out

    def backward(self, dout): #反向传播
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```
代码实例化买两个苹果和三个橘子问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523133505614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
```javascript
apple = 100 #苹果的单价
apple_num = 2 #苹果的数目
orange = 150 #橘子的单价
orange_num=3 #橘子的数目
tax = 1.1 #价格税

# layer
mul_apple_layer = MulLayer() #实例化计算苹果总价层
mul_orange_layer = MulLayer() #实例化计算橘子总价层
add_fruit_layer = AddLayer() #实例化两种水果的总价层
mul_tax_layer = MulLayer() #实例化计算完价格税的总价层

# forward

apple_price = mul_apple_layer.forward(apple, apple_num) #向前传播得出结果
orange_price = mul_orange_layer.forward(orange, orange_num) #向前传播得出结果
fruit_price = add_fruit_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(fruit_price, tax)

#backward

dprice = 1 #正向传播最后的输出对自己的偏导为1
dfruit_price, dtax = mul_tax_layer.backward(dprice) #反向传播求出前一层总价的导数和价格税的导数
dapple_price, dorange_price = add_fruit_layer.backward(dfruit_price) #总价的导数再向前传播求出苹果总价的导数和橘子总价的导数
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  #橘子总价的导数再向前传播求出橘子单价的导数和橘子数目的导数
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #苹果总价的导数再向前传播求出苹果单价的导数和苹果数目的导数

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650
```
# 5.5　激活函数层的实现
## 5.5.1　ReLU 层
公式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523134656151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
对$x$求导：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052313472854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 如果正向传播时的输入 x 大于 0，则反向传播会将上游的值原封不动地传给下游
* 如果正向传播时的 x 小于等于 0，则反向传播中传给下游的信号将停在此处

ReLu层（类）的代码实现：
```javascript
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) #x数组小于等于0的部分在mask里为true，大于0的部分为false
        out = x.copy() #out复制x数组
        out[self.mask] = 0 #x数组所有值小于0即true的都输出0，其余按原本的值输出

        return out

    def backward(self, dout):
        dout[self.mask] = 0 #反向传播，在mask中已经保存了小于等于0的值，直接停止该信号
        dx = dout

        return dx
```
## 5.5.2　Sigmoid 层
公式如下：

$$y=\frac{1}{1+{\rm e}^{-x}}$$
对上述公式求导
$$\frac{\partial{y}}{\partial{x}}={\frac{1}{1+{\rm e}^{-x}}}^2{\rm e}^{-x}=\frac{1}{1+{\rm e}^{-x}}\frac{{\rm e}^{-x}}{1+{\rm e}^{-x}}=y(1-y)$$
* Sigmoid 层的反向传播，只根据正向传播的输出就能计算出来

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523142238296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
```javascript
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x)) #把正向传播输出的结果直接保存到了out中
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out #反向传播用out计算出了结果

        return dx
```
# 5.6　Affine/Softmax 层的实现
## 5.6.1　Affine 层
*神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”{1[几何中，仿射变换包括一次==线性变换==和==一次平移==，分别对应神经网络的加权和运算与加偏置运算。——译者注]}。**因此，这里将进行仿射变换的处理实现为“Affine 层”。*** 
Affine层计算图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523142910696.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
计算图反向传播公式：
$$\frac{\partial{L}}{\partial{W}}=X^T*\frac{\partial{Y}}{\partial{W}}$$
$$\frac{\partial{L}}{\partial{X}}=\frac{\partial{Y}}{\partial{X}}*W^T$$
其中$X^T$和$W^T$是矩阵的转置，相应的维度会交换。
计算图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523143711697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 我们要注意到，$\boldsymbol{X}$ 和 $\frac{\partial L}{\partial\boldsymbol{X}}$ 形状相同，$\boldsymbol{W}$ 和 $\frac{\partial L}{\partial\boldsymbol{W}}$ 形状相同
* 通过以上使矩阵对应维度的元素个数一致的乘积运算法则就可以推导出来反向传播公式
## 5.6.2　批版本的 Affine 层
$N$个数据的Affine层的计算图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/201905231447441.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 把输入变成了$N$维的数据，根据相对应的维度，符合Affine层的计算公式
* 对于偏置，由于输入时会被加到每一个数据（第 1 个、第 2 个……）上。因此，反向传播时，为了保证$\boldsymbol{B}$ 和 $\frac{\partial L}{\partial\boldsymbol{B}}$ 维度的一致，各个数据的反向传播的值需要汇总为偏置的元素。用代码表示的话，如下所示:

```javascript
#forward
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print(X_dot_W)# array([[ 0,  0,  0],
       					[ 10, 10, 10]])
print(X_dot_W + B)# array([[ 1,  2,  3],
      						 [11, 12, 13]])
      						 
# backward
dY = np.array([[1, 2, 3,], [4, 5, 6]])
print(dY)# array([[1, 2, 3],
       [4, 5, 6]])
dB = np.sum(dY, axis=0)
print(dB)# array([5, 7, 9])
```
* 这里使用了 np.sum() 对第 0 轴（以数据为单位的轴，axis=0）方向上的元素进行求和，即对所有数据的每个列分别进行求和。

Affine层实现代码如下：
```javascript
class Affine: #Affine层（类）
    def __init__(self, W, b): #初始化，保存权重偏置，保存输入、权重矩阵的导数、偏置矩阵的导数
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x): #正向传播代入公式
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout): #反向传播
        dx = np.dot(dout, self.W.T) #上游导数和W.T做矩阵乘法
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```
## 5.6.3　Softmax-with-Loss 层
* softmax 函数会将输入值正规化（将输出值的和调整为 1）之后再输出，所以在学习阶段，对于给出答案，调整参数是有价值的。
* 在推理阶段时，是需要给出一个最大值的答案即可，而softmax函数并不改变输出的排序和大小，所以不需要用。

Softmax-with-Loss 层（Softmax 函数和交叉熵误差）的计算图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523150419576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
* 通过$\log$时的计算：$f=log(y1)$，$\frac{\partial{f}}{\partial{y_1}}=\frac{1}{y1}$
* 通过$\log$后的$*$时的其中一个计算：$y1=\frac{1}{S}$，$f=y1=\frac{1}{S}*\rm {e}^{a1}$，
$\frac{\partial{y_1}}{\partial{\frac{1}{S}}}=\rm {e}^{a1}$，
$\frac{\partial{y_1}}{\partial{\frac{1}{S}}}*\frac{-t1}{y1}=-t1S1$
* 通过/时的计算：$f=\frac{1}{S}$，$\frac{\partial{f}}{\partial{S}}=\frac{-1}{s^2}$
$\frac{\partial{f}}{\partial{S}}*((-t1S)+(-t2S)+(-t3S))=1/S$，因为t是one-hot格式，所以和为1
* 通过exp时的计算：$f=\rm {e}^{a1}$，$\frac{\partial{f}}{\partial{a1}}=\rm {e}^{a1}$，$\frac{\partial{f}}{\partial{a1}}*(1/S+-t1/\rm {e}^{a1})=\frac{\rm {e}^{a1}}{S}-t1=y1-t1$



简化图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190523151216587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
*  $(y_1-t_1,y_2-t_2,y_3-t_3)$是 Softmax 层的反向传输的输出和监督标签的差分
* 神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质
* 这个误差表示出了，实际输出和监督标签之间的差距，根据这个误差，可以更有效的检测参数的变化

例子如下：
1. 监督标签是`（0, 1, 0）`，Softmax 层的输出是` (0.3, 0.2, 0.5) `的情形。因为正确解标签处的概率是 0.2（20%），这个时候的神经网络未能进行正确的识别。此时，Softmax 层的反向传播传递的是` (0.3, -0.8, 0.5) `这样一个大的误差。
2. 监督标签是 `(0, 1, 0)`，Softmax 层的输出是 `(0.01, 0.99, 0)` 的情形（这个神经网络识别得相当准确）。此时 Softmax 层的反向传播传递的是` (0.01, -0.01, 0)` 这样一个小的误差。

Softmax-with-Loss 层实现代码如下：
```javascript
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None    # softmax的输出
        self.t = None    # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0] #取出批数据的数量
        dx = (self.y - self.t) / batch_size #计算出单个数据的误差

        return dx 
```
# 5.7　误差反向传播法的实现
## 5.7.1　神经网络学习的全貌图
通过以下两个变化我们来再次实现$2$层神经网络的类：
1. 将 4.5 节中流程图里计算梯度的数值微分方式用误差反向传播法来替代
2. 引入层（类）这种传递的过程
## 5.7.2　对应误差反向传播法的神经网络的实现
实现代码如下：
```javascript
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import * #保存我们之前所有实现了的层的字典变量，例如：layers[Affine],layers[ReLu],layers[Softmax]等
from common.gradient import numerical_gradient #引入数值微分计算梯度的函数（上一章）
from collections import OrderedDict #有序字典，记录传入变量的顺序，更好实现值的传播

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):#初始化时包含输入层的神经元数、隐藏层的神经元数、输出层的神经元数
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size) #高斯分布的随机数进行权重初始化
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict() #将layers的层保存在有序字典里，保证向前传播和向后传播的顺序一致
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1']) #layers类的第一个隐藏层的加权和偏置的结果通过调用Affine（W1，b1）来初始化
        self.layers['Relu1'] = Relu() #layers类的用Relu层作为激活函数的初始化
        self.layers['Affine2'] = \#layers类的第二个隐藏层的加权和偏置的结果通过调用Affine（W2，b2）来初始化
            Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss() #输出层后通过Softmax-with-Loss 层

    def predict(self, x): #识别函数调用每个层的向前传播的函数，并且把输出作为下一个层的输入
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):  #损失函数
        y = self.predict(x) #调用识别函数
        return self.lastLayer.forward(y, t) #每一层向前传播进行输出和标签的损失拟合

    def accuracy(self, x, t): #识别精度函数
        y = self.predict(x) #识别函数
        y = np.argmax(y, axis=1) #取概率最大值
        if t.ndim != 1 : t = np.argmax(t, axis=1) #如果t的维度不为1，取最大值标签
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t): #数值微分梯度函数
        loss_W = lambda W: self.loss(x, t) #调用损失函数

        grads = {} #grads保存梯度的字典型变量
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads 返回梯度值

    def gradient(self, x, t): #用误差反向传播法求梯度函数
        # forward
        self.loss(x, t)  #调用损失函数向前传播
        # backward
        dout = 1 #输出导数为1
        dout = self.lastLayer.backward(dout) #最后一层反向传播求输出

        layers = list(self.layers.values()) #用另一个layers类来保存反向调用的层
        layers.reverse() #layers数组翻转
        for layer in layers:
            dout = layer.backward(dout) #反向传播函数的调用顺序会改变

        #用grads保存梯度的字典型变量
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW 
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
```
## 5.7.3　误差反向传播法的梯度确认
* 数值积分：实现简单，不易出错，确认误差反向传播法的实现是否正确
* 误差反向传播法：实现很复杂，容易出错
* 确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致的操作称为梯度确认（gradient check）。

梯度确认代码如下：
```javascript
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = \ load_mnist(normalize=True, one_
hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 训练数据和监督标签的保存
x_batch = x_train[:3]
t_batch = t_train[:3]
# 调用
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
```
运行结果如下：
b1:9.70418809871e-13
W2:8.41139039497e-13
b2:1.1945999745e-10  
## 5.7.4　使用误差反向传播法的学习
```javascript
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 #学习重复次数
train_size = x_train.shape[0]
batch_size = 100 #batch选择的数据数量
learning_rate = 0.1 #学习率
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) #决定epoch的值

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #10000个里选100个的数据位置放入batch_mask
    x_batch = x_train[batch_mask] #把这些位置上的输入数据取出来放入x_batch作为mini—batch学习的输入数据
    t_batch = t_train[batch_mask] #把这些位置上的监督数据取出来放入t_batch作为mini—batch学习的监督数据

    # 通过误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] #通过减去梯度和学习率的乘积，减小损失函数的值

    loss = network.loss(x_batch, t_batch) #计算损失函数
    train_loss_list.append(loss) #每次的损失函数加入到数组中
	# 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

end
* 原书为《深度学习入门 基于Python的理论与实现》作者：斋藤康毅    
人民邮电出版社
* 本文章是gitchat的《陆宇杰的训练营：15天共读深度学习》[^1]的课程读书笔记
* 本文章大量引用原书中的内容和训练营课程中的内容作为笔记

[^1]:[《陆宇杰的训练营：15天共读深度学习》](https://gitbook.cn/gitchat/column/5cc2b3afd575995386d3abc7)
