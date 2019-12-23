#导入所有的依赖包
import  tensorflow as tf
import numpy as np
import os
from PIL import Image
import csv

def create_model():
    model_name = r'.\model_dir\safety\cnn_model.h5'
    # 判断是否已经有model文件存在，如果model文件存在则加载原来的model并在原来的moldel继续训练，如果不存在则新建model相关文件
    if os.path.exists(model_name):
        # 如果存在模型文件，则加载存放model文件夹中最新的文件
        print("Reading model parameters from %s" % model_name)
        # 使用tf.keras.models.load_model来加载模型
        model = tf.keras.models.load_model(model_name)
        return model
    else:
        print("There is no model")
        exit(0)

# 定义预测函数
def predict(model, data):
    # 使用predict方法对输入数据进行预测
    prediction = model.predict(data)
    # 返回预测结果
    return prediction[0][0]

if __name__ == "__main__":
    # 初始化model
    model = create_model()
    image_path = os.path.abspath(r'.\_SpatialData\GIS-Q4\03\20191212_100933\Image_20191212_100933')
    # image_path = os.path.abspath(r'E:\_SpatialData\GIS-Q4\03\20191212_100933\test_image\safety\test')
    output_data = os.path.abspath(r'.\_SpatialData\GIS-Q4\03\20191212_100933\safety.csv')
    # output_data = os.path.abspath(r'E:\_SpatialData\GIS-Q4\03\20191212_100933\test_image\safety\safety_test.csv')
    dirs = os.listdir(image_path)
    data = [['name', 'score']]
    for f in dirs:
        filename = os.path.join(image_path, f)
        # 使用PIL中的Image打开文件并获取图像文件中的信息
        img = Image.open(filename)
        # 缩放图片的尺寸
        img = img.resize((480, 300))
        # 将图像文件的格式转换为RGB
        img = img.convert("RGB")
        # 分别获取r,g,b三元组的像素数据并进行拼接
        r, g, b = img.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        img = np.concatenate((r_arr, g_arr, b_arr))
        # 将拼接得到的数据按照模型输入维度需要转换为（300, 480, 3)，并对数据进行归一化
        image = img.reshape([1, 300, 480, 3])/255
        # 调用predict方法进行预测
        result = predict(model, image)
        print(result)
        data.append([f, result])

    # 输出成csv文件
    with open(output_data, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
