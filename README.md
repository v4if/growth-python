### growth-python
增长中的python程序库

### List
> calc-interpreter

可以加括号，任意数量的加减乘除运算解析器
![growth-python](https://github.com/v4if/growth-python/raw/master/calc-interpreter/2016-10-23-143524.png)

> k-mean

k-mean聚类算法的数据挖掘与图像分割

K-mean.py 

k-mean聚类算法数据挖掘，将数据进行聚类分析，实验数据为data.txt

`^`为每个簇cluster的质心

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-25-220155.png)

K-mean2.py

k-mean聚类算法图像分割，将图像分割并着色，实验数据为favorite.png

TODO:

    请确定已经安装了python-opencv

    sudo apt-get install python-opencv 

原图像

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-25-131937.png)

K = 5, iterator = 1

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-24-212419_5-1.png)

K = 10, iterator = 2

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-25-142024_10-2.png)

K-mean3.py

添加了像素点的坐标作为一个样本的特征值，坐标所占的权重为0.5，可设置

K = 10, iterator = 2, coord_K = 0.5

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-25-131556_C10-5.png)

K = 100, iterator = 10, coord_K = 0.1

![growth-python](https://github.com/v4if/growth-python/raw/master/k-mean/testout/2016-10-25-134935_C100-10.png)

> md5-explode

md5在线爆破解密
代码用的在线网站数据库检查修复中，而且添加了验证码，原来的程序已经跑不出来密码了。。。

> showcode

统计项目根目录下代码行数
![growth-python](https://github.com/v4if/growth-python/raw/master/showcode/2016-10-23-144952.png)

> zip-explode

多线程本地字典爆破zip密码
![growth-python](https://github.com/v4if/growth-python/raw/master/zip-explode/2016-10-23-150234.png)
