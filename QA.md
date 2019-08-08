## 激活函数的作用：非线性化

如果我们不运用激活函数的话，则输出信号将仅仅是一个简单的线性函数。线性函数一个一级多项式。现如今，线性方程是很容易解决的，但是它们的复杂性有限，并且从数据中学习复杂函数映射的能力更小。一个没有激活函数的神经网络将只不过是一个线性回归模型（Linear regression Model）罢了，它功率有限，并且大多数情况下执行得并不好。我们希望我们的神经网络不仅仅可以学习和计算线性函数，而且还要比这复杂得多。同样是因为没有激活函数，我们的神经网络将无法学习和模拟其他复杂类型的数据，例如图像、视频、音频、语音等。这就是为什么我们要使用人工神经网络技术，诸如深度学习（Deep learning），来理解一些复杂的事情，一些相互之间具有很多隐藏层的非线性问题，而这也可以帮助我们了解复杂的数据。

那么为什么我们需要非线性函数？

非线性函数是那些一级以上的函数，而且当绘制非线性函数时它们具有曲率。现在我们需要一个可以学习和表示几乎任何东西的神经网络模型，以及可以将输入映射到输出的任意复杂函数。神经网络被认为是通用函数近似器（Universal Function Approximators）。这意味着他们可以计算和学习任何函数。几乎我们可以想到的任何过程都可以表示为神经网络中的函数计算。

而这一切都归结于这一点，我们需要应用激活函数f（x），以便使网络更加强大，增加它的能力，使它可以学习复杂的事物，复杂的表单数据，以及表示输入输出之间非线性的复杂的任意函数映射。因此，使用非线性激活函数，我们便能够从输入输出之间生成非线性映射。

激活函数的另一个重要特征是：它应该是可以区分的。我们需要这样做，以便在网络中向后推进以计算相对于权重的误差（丢失）梯度时执行反向优化策略，然后相应地使用梯度下降或任何其他优化技术优化权重以减少误差。

只要永远记住要做：

“输入时间权重，添加偏差和激活函数”

最流行的激活函数类型

1.Sigmoid函数或者Logistic函数

2.Tanh — Hyperbolic tangent（双曲正切函数）

3.ReLu -Rectified linear units（线性修正单元）

## 1.Eager API有什么用?

## 2.除了梯度下降法，是否还有其他求最小误差方法？

误差模型常用的有：
```
//最小平方差
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
```
```
//最小交叉熵
cross_entrop = -tf.reduce_sum(y_*tf.log(y))
```
```
//梯度下降法（高斯-牛顿法等）
tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy或cost)
```
## 3.softmax的介绍：https://blog.csdn.net/wgj99991111/article/details/83586508

Softmax的含义：Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显。

对每个类别分别计算出一个值，再由指数函数e^x进行概率的转化

## 4. nn介绍：近邻算法，通过单个测试样本与一组测试样本进行比较，得到误差最小的结果 作为测试样本预测结果。
https://blog.csdn.net/margretwg/article/details/64132791

## 5. k-means介绍：通过设置k个质心，计算各个sample与质心的距离进行分类。再计算平均值作为质心。重复上述步骤直到收敛或符合预置条件
https://www.cnblogs.com/en-heng/p/5173704.html

## 4.tf.reduce_mean ,沿指定轴的方向计算均值，0按行，1按列。 参考：
https://blog.csdn.net/dcrmg/article/details/79797826

## 5.tf.reduce_sum,,reduce_sum应该理解为压缩求和，用于降维

按照行或列求和都是得到1xN的输出，按行的维度求和得到Nx1的输出
```
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
#求和
tf.reduce_sum(x) ==> 6
#按列求和
tf.reduce_sum(x, 0) ==> [2, 2, 2]
#按行求和
tf.reduce_sum(x, 1) ==> [3, 3]
#按照行的维度求和
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
#行列求和
tf.reduce_sum(x, [0, 1]) ==> 6
```

## 5.tf.argmin和argmax类似，按argmax为例：

在tf.argmax( , )中有两个参数，第一个参数是矩阵，第二个参数是0或者1。0表示的是按列比较返回最大值的索引，1表示按行比较返回最大值的索引。
```
import tensorflow as tf

Vector = [1,1,2,5,3]           #定义一个向量
X = [[1,3,2],[2,5,8],[7,5,9]]  #定义一个矩阵

with tf.Session() as sess:
    a = tf.argmax(Vector, 0)
    b = tf.argmax(X, 0)
    c = tf.argmax(X, 1)

    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
```
输出结果:
```
[2 1 2] //第一列第三，第二列第二，第三列第三个数最大
[1 2 2] //第一行第二，第二行第三，第三行第三个数最大
```
## 6. openCV的py库， import的时候是import cv2 ？

## 7. imutils是个什么库？
## 8. cv2.dnn.blobFromImage(src,scal,size,mean-declare)  (1,3,244,244) 
创建一个blob数据类型，用于caffe模型。blob类似与tensorflow里面的tensor

## 9.cv2里面的一些函数方法问题,用来进行图像处理

1.灰度图
```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
2.canny边缘检测图
```
edged = cv2.Canny(gray, 30, 160)
```
3.阀值处理图像
```
//cv2.THRESH_BINARY_INV表示如何根据阀值进行取值，二值化
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
```
4.绘制等高线图

参考网站：https://blog.csdn.net/keith_bb/article/details/70185209
```
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)
```
5.侵蚀 或 膨胀处理噪声
```
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)

mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
```
6.图像切片
```
roi = image[60:160, 320:420]
```
7.图像缩放
```
//无视宽高比
resized = cv2.resize(image, (200, 200))
//宽高比不变
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
//使用imutils宽高比不变
resized = imutils.resize(image, width=300)
```
8.图像旋转
```
//绕中心点，顺时针旋转45度
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0) # 1.0 means scale value
rotated = cv2.warpAffine(image, M, (w, h))
//同上，图片可以被切割
rotated = imutils.rotate(image, -45)
//旋转，且图片保持完整性
rotated = imutils.rotate_bound(image, 45)
```
9.高斯模糊效果
```
//11x11代表使用的高斯核
blurred = cv2.GaussianBlur(image, (11, 11), 0) # 0 可以代表模糊权重
```
10.在图像上画方、圆、线、字
```
//rectangle
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
//-1 代表画实心
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
//line
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
//text (10,25)代表启示位置，文字左下点坐标
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```