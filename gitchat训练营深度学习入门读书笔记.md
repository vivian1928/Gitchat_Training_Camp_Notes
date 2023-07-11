@[TOC](第一章 Python 入门)

# Python简介
自从20世纪90年代初Python语言诞生至今，它已被逐渐广泛应用于系统管理任务的处理和Web编程。
## 主要目的
* 用于搭建深度学习的神经网络和卷积神经网络
* 用于写python的网络爬虫爬取网页图片数据
* 用于图像处理
* 用于对个人学习有更深入的实践的理解
# Python的安装
## Python的版本
Python3.7  下载网址：[Python官网](https://www.python.org/downloads/release/python-373/)
## mac手动安装流程
![出现python安装器](https://img-blog.csdnimg.cn/20190516150332573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
 1. **mac自带版本** ，python2.7；
 2. **如何切换版本**，设置python的环境变量路径：
	* vi ~/.bash_profile查看环境变量文件
	* i切换入编辑模式
	* 切换python的环境变量
	* exit切换命令模式，`：`切换，`wq`保存并退出
	* source ～/.bash_profile

```javascript
alias python2='/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7'
alias python3='/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6'
alias python=python3
```
  
  
 4.  **查看当前python版本以及python解释器位置**:	`python  --version`和 `which python`；

## anaconda安装
anaconda下载网址：[anaconda官网](https://www.anaconda.com/distribution/#download-section)
下载好点击同意或者下一步直接安装下去即可
安装好后，打开
![anaconda图标](https://img-blog.csdnimg.cn/20190516165201542.png)
可以看到home界面
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516165235931.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
但这时在Terminal输入`conda`指令是没有用的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516165506526.png)
* 配置anaconda环境变量
	终端中打开环境变量文件`vi ~/.bash_profile`，写入
	```javascript
	export PATH="/usr/anaconda3/bin:$PATH"
	```
	就会得到下面的结果
	```javascript
	usage: conda [-h] [-V] command ...
	conda is a tool for managing and deploying applications, environments and packages.
	Options:positional arguments:
  command
    clean        Remove unused packages and caches.
    config       Modify configuration values in .condarc. This is modeled
                 after the git config command. Writes to the user .condarc
                 file (/Users/mac/.condarc) by default.
    create       Create a new conda environment from a list of specified
                 packages.
    help         Displays a list of available conda commands and their help
                 strings.
     info         Display information about current conda install.
    init         Initialize conda for shell interaction. [Experimental]
    install      Installs a list of packages into a specified conda
                 environment.
    list         List linked packages in a conda environment.
    package      Low-level conda package utility. (EXPERIMENTAL)
    remove       Remove a list of packages from a specified conda environment.
    uninstall    Alias for conda remove.
    run          Run an executable in a conda environment. [Experimental]
    search       Search for packages and display associated information. The
                 input is a MatchSpec, a query language for conda packages.
                 See examples below.
    update       Updates conda packages to the latest compatible version.
    upgrade      Alias for conda update.
    optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.
	conda commands available from other packages:
  build
  convert
  debug
  develop
  env
  index
  inspect
  metapackage
  render
  server
  skeleton
  verify
	```
这时进入了base环境，即anaconda搭建时创造的环境
* 创建指定python版本的环境：
conda命令create 创建，-n （可以省略）指定名称，python=指定python版本，packages需要用到的包。
```javascript
conda create -n env_name list of packages 
```
* 进入环境
```javascript
source activate env_name
```
* 退出环境
```javascript
source deactivate
```
* 显示所有的环境	
```javascript
conda env remove -n env_name
```
回到anaconda界面，点击environment，可以看到安装好的所有配置
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516172739419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
已经有了两个必要的库`Numpy`和`Matplotlib`

## anaconda两个自带的库

* Numpy用来数值计算
* Matplotlib 用来画图，实现数据可视化
* anaconda侧重于数据分析



## Python解释器的查看
* Terminal输入`python  --version`可以查看到版本如下
	

```
Python 3.7.3
```

* 输入`python` ,可以打开python解释器：python解释器打开后，可以直接在终端编程
	

```
Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 16:52:21) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

### 解释器中对话方式编程实例
1.  算术计算
 ```javascript
 >>> 1+2
3
>>> 6*6*6
216
>>> 7/5
1.4
>>> 3**3
27
 ```
 * 但在 Python 3.x 中，整数除以整数的结果是小数（浮点数）。
2. 数据类型
 * Python 中的 type() 函数可以用来查看数据类型。
 ```javascript
 >>> type(1)
<class 'int'>
>>> type(2.2)
<class 'float'>
>>> type("Hello World")
<class 'str'>
 ```
 * 注意字符串类型需要加双引号表示
3. 变量
```javascript
>>> x=10 #初始化
>>> print(x)
10
>>> x=3.14		#赋值
>>> print(x)
3.14
>>> y=20
>>> x*y
62.800000000000004
>>> type(x*y)
<class 'float'>
```
* python是动态判断变量类型的
* `#`可以用来做行注释
4. 列表
```javascript
>>> a=[1,2,3]  #生成数组
>>> print(a)
[1, 2, 3]
>>> len(a)
3
>>> a[0]
1
>>> a[1]
2
>>> a[2]=99  #赋值
>>> print(a)
[1, 2, 99]
```
* Python 的列表提供了切片（slicing）这一便捷的标记法。
```javascript
>>> b=[1,2,3,4,5,6,7,8,9]
>>> b[0:5]
[1, 2, 3, 4, 5]
>>> b[:3]
[1, 2, 3]
>>> b[4:]
[5, 6, 7, 8, 9]
>>> b[:-4]
[1, 2, 3, 4, 5]
>>> b[:-1]
[1, 2, 3, 4, 5, 6, 7, 8]
```
5. 字典
* 字典以键值对保存数据
```javascript
>>> dic={"hello":180}
>>> dic["hello"]
180
>>> dic["world"]=170
>>> print(dic)
{'hello': 180, 'world': 170}
```
6. 布尔型
```javascript
>>> sunny=True
>>> rain=False
>>> type(sunny)
<class 'bool'>
>>> not rain
True
>>> rain and sunny
False
```
7. if语句
```javascript
>>> if sunny:
...     print("Nice day")
... else:
...     print("Rainy day")
... 
Nice day
```
*  当出现expected an indented block异常时，表示要代码缩进，python需要严格的代码缩进格式,缩进可以用`tab`，也可以用4个空白字符。

8. for语句
```javascript
>>> for j in b:
...     print(j)
... 
1
2
3
4
5
6
7
8
9
```
9. 函数
```javascript
>>> def hello():
...     print("Hello World!")
... 
>>> hello()
Hello World!
```
* 字符串的拼接可以使用 +。
关闭python解释器时使用exit()或ctrl+d即可
## Python脚本文件创建
在文本编辑器新建一个 hungry.py 的文件。hungry.py 只包含下面一行语句。
`print("I'm hungry!")`
必须要用文本编辑器，否则会出现以下错误
`can't find '__main__' module in 'hungry.py'`
接着，打开Terminal，移至 hungry.py 所在的位置。然后，将 hungry.py 文件名作为参数，运行 python 命令。这里假设 hungry.py 在~/Document 目录下
```javascript 
cd ~/Documents # 移动目录
Documents mac$ python hungry.py
I'm hungry!
```
## 类
下面我们通过一个简单的例子来创建一个类。这里将下面的程序保存为 man.py(在jupiter搭建的环境下)。

    class Man:
        def __init__(self, name):
            self.name = name
            print("Initialized!")
    
        def hello(self):
            print("Hello " + self.name + "!")
    
        def goodbye(self):
            print("Good-bye " + self.name + "!")
    
        m = Man("David")
        m.hello()
        m.goodbye()

从终端运行 man.py。

    $ python man.py
    Initialized!
    Hello David!
    Good-bye David!

# Numpy
## 导入 NumPy生成数组
* 先用Jupiter notebook搭建了环境

```javascript
>>> import numpy as np
>>> x=np.array([1.0,2.0,3.0])
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
```
## Numpy算术运算（element-wise）
### 数组 x 和数组 y 的元素个数是相同的
```javascript
>>> y=np.array([2.0,5.0,1.0])
>>> x+y
array([3., 7., 4.])
>>> x*y
array([ 2., 10.,  3.])
>>> x/y
array([0.5, 0.4, 3. ])
```
### 单一的数值（标量）
```javascript
>>> x/2
array([0.5, 1. , 1.5])
```
## Numpy的N维数组
```javascript
>>> A=np.array([[3,4],[1,5]])
>>> print(A)
[[3 4]
 [1 5]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int64')
```
每一个`[]`代表一维的数组
* N维数组的算术运算
```javascript
>>> B = np.array([[3, 0],[0, 6]])
>>> A+B
array([[ 6,  4],
       [ 1, 11]])
>>> A*B
array([[ 9,  0],
       [ 0, 30]])
```
1. 数学上将一维数组称为向量，将二维数组称为矩阵。
2. 可以将一般化之后的向量或矩阵等统称为张量（tensor）。
3. 本书基本上将二维数组称为“矩阵”，将三维数组及三维以上的数组称为“张量”或“多维数组”。
* 不同维度的数组之间的算术运算运用了广播的功能
```javascript
>>> A*10
array([[30, 40],
       [10, 50]])
```

## 广播
在上例中标量 10 被扩展成了 2 × 2 的形状，然后再与矩阵 A 进行乘法运算。这个巧妙的功能称为广播（broadcast）。
```javascript
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([10, 20])
>>> A * B
array([[ 10, 40],
       [ 30, 80]])
```
![广播的例子](https://img-blog.csdnimg.cn/20190516224831985.png)
图 1-2　广播的例子 2[^1]

[^2]: 注脚的解释


##  访问元素
1. 元素访问
```javascript
>>> A[0]
array([3, 4])
>>> A[1][0]
1
```
2. `for in`语句访问每行元素
```javascript
>>> for row in A:
...     print(row)
... 
[3 4]
[1 5]
>>> 
```
3. 数组访问指定元素
```javascript
>>> A=A.flatten()
>>> print(A)
[3 4 1 5]
>>> A[np.array([0,1,3])]	#获取一维数组中索引为0，1，3的元素
array([3, 4, 5])
>>> A>2		#判断数组中满足条件的元素
array([ True,  True, False,  True])
>>> A[A>2]		#数组[condition]表示取满足条件的元素
array([3, 4, 5])
```
* 对 NumPy 数组使用不等号运算符等（上例中是 X > 15），结果会得到一个布尔型的数组。上例中就是使用这个布尔型数组取出了数组的各个元素（取出 True 对应的元素）。

*[HTML]:   超文本标记语言

# Matplotlib
## 绘制简单图形
```javascript
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 10, 1.0) # 以1.0为单位，生成0到10的数据
y = np.sin(x) # 对x中的每个数据求sin()值

# 绘制图形
plt.plot(x, y) # 绘制图形
plt.show() # 显示图形
```
* 注意加上%maplotlib inline ，可以在Jupiter Notebook中调用show()显示出来
绘制图形如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516232002639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)
## Pyplot功能

```javascript
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin") # label曲线的标签名
plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend() # 显示图中的标签
plt.show()
```
绘制图形如下：
![- 关于 **甘特图** 语法，参考 \[这儿\]\[2\],](https://img-blog.csdnimg.cn/20190516232638269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)

## 显示图像

```javascript
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.image import imread # 从A包中引入B函数语句
img = imread('/Users/mac/Documents/lena.png') # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()
```
* pyplot 中还提供了用于显示图像的方法 imshow()。另外，可以使用 matplotlib.image 模块的 imread() 方法读入图像。
显示图像如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516233244443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzExNDg4NQ==,size_16,color_FFFFFF,t_70)


