import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# 读取数据集
mnist=np.load(r'./data/mnist_data/mnist.npz')

# 划分训练集,测试集
x_train=mnist['x_train']
x_test=mnist['x_test']
y_train=mnist['y_train']
y_test=mnist['y_test']

# 将图像数据标准化到[0, 1]范围内
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 对标签进行独热编码（One-Hot Encoding）
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 将二维图像展平为一维数组
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# 使用支持向量机（SVM）进行训练
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(x_train_flat,np.argmax(y_train, axis=1))

# 预测并评估模型
y_pred = svm_classifier.predict(x_test_flat)
accuracy = accuracy_score(np.argmax(y_test,axis=1), y_pred)
print("SVM模型的准确度：", accuracy)