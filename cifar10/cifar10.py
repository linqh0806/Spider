# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
#!/usr/bin/python
# coding:utf-8

# 建立CIFAR-10的模型
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

# 描述CIFAR-10数据集的全局常量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# 描述训练过程的常量
MOVING_AVERAGE_DECAY = 0.9999     # 滑动平均衰减率.
NUM_EPOCHS_PER_DECAY = 350.0      # 在学习速度衰退之后的Epochs.
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习速率衰减因子.
INITIAL_LEARNING_RATE = 0.1       # 初始学习率.

# 如果模型使用多个GPU进行训练,则使用tower_name将所有Op名称加前缀以区分操作
# 可视化模型时从摘要名称中删除此前缀
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 激活摘要创建助手
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # 若多个GPU训练,则从名称中删除'tower_[0-9]/',利于TensorBoard显示
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  # 提供激活直方图的summary
  tf.summary.histogram(tensor_name + '/activations', x)
  # 衡量激活稀疏性的summary
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# 创建存储在CPU内存上的变量(变量的名称,整数列表,变量的初始化程序)
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

# 创建一个权重衰减的初始化变量(变量的名称,整数列表,截断高斯的标准差,加L2Loss权重衰减)
# 变量用截断正态分布初始化的.只有指定时才添加权重衰减
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  # 用截断正态分布进行初始化
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    # wd用于向losses添加L2正则化,防止过拟合,提高泛化能力
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    # 把变量放入一个集合
    tf.add_to_collection('losses', weight_decay)
  return var

# -------------------------模型输入-----------------------------------
# 训练输入
# 返回：images:[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]; labels:[batch_size]
def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  # 读入并增广数据
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)

# 预测输入
# 返回：images:[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]; labels:[batch_size]
def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  # 图像预处理及输入
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)
# -------------------------------------------------------------------


# -------------------------模型预测-----------------------------------
# 构建CIFAR-10模型
# 使用tf.get_variable()而不是tf.Variable()来实例化所有变量,以便跨多个GPU训练运行共享变量
# 若只在单个GPU上运行,则可通过tf.Variable()替换tf.get_variable()的所有实例来简化此功能
def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  # 卷积层1
  with tf.variable_scope('conv1') as scope:
    # weight不进行L2正则化
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    # 卷积
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # biases初始化为0
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    # 卷积层1的结果由ReLu激活
    conv1 = tf.nn.relu(bias, name=scope.name)
    # 汇总
    _activation_summary(conv1)

  # pool1
  # 池化层1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  # lrn层1　局部响应归一化:增强大的抑制小的,增强泛化能力
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  # 卷积层2
  with tf.variable_scope('conv2') as scope:
    # weight不进行L2正则化
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    # biases初始化为0.1
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    # 卷积层2的结果由ReLu激活
    conv2 = tf.nn.relu(bias, name=scope.name)
    # 汇总
    _activation_summary(conv2)

  # norm2
  # lrn层2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  # 池化层2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  # 全连接层3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
      # 维数
      dim *= d
    # 将样本转换为一维向量
    reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
    # 添加L2正则化约束,防止过拟合
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    # biases初始化为0.1
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    # ReLu激活
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  # 全连接层4
  with tf.variable_scope('local4') as scope:
    # 添加L2正则化约束,防止过拟合
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    # biases初始化为0.1
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    # ReLu激活
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  # 线性层
  # (WX+b)不使用softmax,因为tf.nn.sparse_softmax_cross_entropy_with_logits接受未缩放的logits并在内部执行softmax以提高效率
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    # biases初始化为0
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    # (WX+b) 进行线性变换以输出 logits
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    # 汇总
    _activation_summary(softmax_linear)

  return softmax_linear
# -------------------------------------------------------------------


# -------------------------模型训练-----------------------------------
# 将L2损失添加到所有可训练变量
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    # 计算logits和labels之间的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    # 计算整个批次的平均交叉熵损失
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 把变量放入一个集合
    tf.add_to_collection('losses', cross_entropy_mean)
    # 总损失定义为交叉熵损失加上所有的权重衰减项(L2损失)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 添加损失的summary;计算所有单个损失的移动均值和总损失
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  # 指数移动平均
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  # 将指数移动平均应用于单个损失
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  # 单个损失损失和全部损失的标量summary
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    # 将每个损失命名为raw,并将损失的移动平均命名为原始损失
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

# 训练CIFAR-10模型
# 创建一个优化器并应用于所有可训练变量,为所有可训练变量添加移动均值(全部损失,训练步数)
def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  # 影响学习率的变量
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  # 指数衰减学习率
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  # 对总损失进行移动平均
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  # 计算梯度
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  # 应用处理过后的梯度
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  # 为可训练变量添加直方图
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  # 为梯度添加直方图
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  # 跟踪所有可训练变量的移动均值
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  # 使用默认图形的包装器
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

# -------------------------------------------------------------------

# 下载并解压数据
def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
