import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# # 加载MNIST数据集
# mnist = tf.keras.datasets.mnist

# # 加载训练集和测试集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 输出训练集和测试集的维度信息
# print("训练集特征维度：", x_train.shape)
# print("训练集标签维度：", y_train.shape)
# print("测试集特征维度：", x_test.shape)
# print("测试集标签维度：", y_test.shape)

mnist=np.load(r'./data/mnist_data/mnist.npz')
# X,y=mnist.data,mnist.target.astype(np.int)

x_train=mnist['x_train']
x_test=mnist['x_test']
y_train=mnist['y_train']
y_test=mnist['y_test']

# # 数据探索
# print("数据集大小:", X.shape)
# print("特征维度:", X.shape[1])
# print("类别数:", len(np.unique(y)))

# print("array_list:", mnist.files)
#
# x_train_shape=mnist['x_train'].shape
# x_test_shape=mnist['x_test'].shape
# y_train_shape=mnist['y_train'].shape
# y_test_shape=mnist['y_test'].shape

# 将图像数据标准化到[0, 1]范围内
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 对标签进行独热编码（One-Hot Encoding）
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("x_train_shape：",x_train.shape)
print("x_test_shape：",x_test.shape)
print("y_train_shape：",y_train.shape)
print("y_test_shape：",y_test.shape)

# 可视化数据集中的部分图像
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title("Label: {}".format(y_train[i]))
    plt.axis('off')
plt.show()

# 类别分布直方图
y_combined = np.concatenate([y_train, y_test])
plt.figure(figsize=(8, 6))
plt.hist(y_combined, bins=np.arange(11) - 0.5, rwidth=0.8, alpha=0.75)
plt.xticks(np.arange(10))
plt.xlabel("类别")
plt.ylabel("样本数量")
plt.title("类别分布")
plt.show()