#!/usr/bin/python
# coding:utf-8

# 读取本地CIFAR-10的二进制文件

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import tensorflow as tf

# 处理图像尺寸,与CIFAR原始图像大小32 x 32不同
IMAGE_SIZE = 24
# 全局常量
NUM_CLASSES = 10
# 训练实例个数
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000#50000
# 验证实例个数
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000#10000

# 读取二进制CIFAR10数据(filename_queue：要读取的文件名)
def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    # CIFAR-10数据集中图像的尺寸
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 每条记录都包含一个标签,后面跟着图像,每个记录都有固定的字节数
    record_bytes = label_bytes + image_bytes
    # 从文件输出固定长度的字段(每个图片的存储字节数是固定的)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 返回reader生成的下一条记录(key, value pair)
    result.key, value = reader.read(filename_queue)
    # 将字符串转换为uint8类型的向量
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 将标签从uint8转换为int32
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 标签之后的字节表示图像,将其从[depth*height*width]转换为[depth,height,width]
    depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
    # 从[depth,height,width]转换为[height,width,depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

# 构建[images,labels]的队列
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    # 使用16个独立线程,16个线程被连续的安排在一个队列中
    # 每次在执行读取一个 batch_size数量的样本[images,labels]
    num_preprocess_threads = 16
    # 是否随机打乱队列
    if shuffle:
        # images:4D张量[batch_size, height, width, 3]; labels:[batch_size]大小的1D张量
        # 将队列中数据打乱后取出
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,                         # 每批次的图像数量
            num_threads=num_preprocess_threads,            # 入队tensor_list的线程数量
            capacity=min_queue_examples + 3 * batch_size,  # 队列中元素的最大数量
            min_after_dequeue=min_queue_examples)          # 提供批次示例的队列中保留的最小样本数
    else:
        # 将队列中数据按顺序取出
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,                         # 从队列中提取的新批量大小
            num_threads=num_preprocess_threads,            # 排列“tensor”的线程数量
            capacity=min_queue_examples + 3 * batch_size)  # 队列中元素的最大数量
    # 在TensorBoard中显示训练图像
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])



# 读入并增广数据为训练构建输入(CIFAR-10数据的路径,每批次的图像数量)
# 返回值　images：[batch_size,IMAGE_SIZE,IMAGE_SIZE,3];labels：[batch_size]
def distorted_inputs(data_dir, batch_size):
    # 获取5个二进制文件所在路径
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 创建一个文件名的队列
    filename_queue = tf.train.string_input_producer(filenames)
    with tf.name_scope('data_augmentation'):
        # 读取文件名队列中的文件
        read_input = read_cifar10(filename_queue)
        # 转换张量类型
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 随机裁剪[height, width]大小的图像
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        # 随机水平翻转图像
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # 随机调整图像亮度与对比度(不可交换)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        # 减去均值除以方差,线性缩放为零均值的单位范数:白化/标准化处理
        float_image = tf.image.per_image_standardization(distorted_image)
        # 设置张量的形状
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])
        # 确保随机乱序具有良好的混合性能
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
        print ('Filling queue with %d CIFAR images before starting to train.'
               'This will take a few minutes.' % min_queue_examples)
    # 构建图像和标签的队列
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


# 图像预处理并为CIFAR预测构建输入
# 输入：(指示是否应该使用训练或eval数据集,CIFAR-10数据的路径,每批次的图像数量)
# 输出：images:[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]; labels: [batch_size]
def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)
    with tf.name_scope('input'):
        # 创建一个生成要读取的文件名的队列
        filename_queue = tf.train.string_input_producer(filenames)
        # 阅读文件名队列中文件的示例
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 用于评估的图像处理
        # 在图像的中心裁剪[height, width]大小的图像,裁剪中央区域用于评估
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
        # 减去平均值并除以像素的方差,保证数据均值为0,方差为1
        float_image = tf.image.per_image_standardization(resized_image)
        # 设置张量的形状
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])
        # 确保随机乱序具有良好的混合性能
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
    # 通过建立一个示例队列来生成一批图像和标签
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
