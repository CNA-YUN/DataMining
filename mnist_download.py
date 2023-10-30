import urllib.request
import os

# 定义保存文件的目录和文件名
save_dir = "./data/mnist_data"
file_name = "mnist.npz"

# 如果保存文件的目录不存在，则创建目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 下载MNIST数据集文件
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
urllib.request.urlretrieve(url, os.path.join(save_dir, file_name))

print("MNIST数据集已下载到本地目录：", os.path.abspath(save_dir))
print("文件名：", file_name)
