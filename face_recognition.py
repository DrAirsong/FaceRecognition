import tensorflow as tf
import random
import types
import os
import numpy

TRAIN_FIEL = 'face_data_set/MTFL/training.txt'  #训练集路径
TEST_FIEL = 'face_data_set/MTFL/testing.txt'    #测试集路径
SAVE_PATH = 'model'     #保存模型的路径

DATA_SIZE = 2000    #此次训练使用的数据项总数量
VALIDAION_SZIE = 25     #验证数据项数量
BATCH_SIZE = 30     #训练时每一批数据数量
EPOCHS = 50     #此次训练轮数
best_validation_loss = 30   #自定义的损失值界限，损失函数值低于此界限的模型才能被保存

#生成weight变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#生成biase变量
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#卷积层函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')

#池化层函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义占位符
#x是实际数据输入
#y_是理想数据输出
#keep_prob是drop out 时的数据保有率
x = tf.placeholder(tf.float32, shape=[None,250,250,1])/255.
y_ = tf.placeholder(tf.float32, shape=[None, 14]) 
keep_prob = tf.placeholder(tf.float32)

#卷积模型函数
def model():
    #卷积层+池化层
    w_conv1 = weight_variable([3,3,1,32])   #设置变量（权重）
    b_conv1 = bias_variable([32])       #设置变量（偏移）

    h_conv1 = tf.nn.relu(conv2d(x, w_conv1)+b_conv1)    #卷积操作+激活操作
    h_pool1 = max_pool_2x2(h_conv1)     #池化操作

    #卷积层+池化层（细节同上）
    w_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #卷积层+池化层（细节同上）
    w_conv3 = weight_variable([2,2,64,128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3)+b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #全连接层
    w_fc1 = weight_variable([30*30*128,500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool3, [-1,30*30*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,w_fc1)+b_fc1)

    #dropout
    w_fc2 = weight_variable([500,500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2,rate=1-keep_prob)

    #输出层
    w_fc3 = weight_variable([500,14])
    b_fc3 = bias_variable([14])

    prediction = tf.matmul(h_fc2_drop, w_fc3)+b_fc3   #获得预测数据输出值

    #使用平方差损失函数计算 预测输出 与 理想输出 的误差
    error = tf.sqrt(tf.reduce_mean(tf.square(y_-prediction)))    
    
    #返回prediction和rmse的声明
    return prediction, error


#载入数据函数
def input_data():
    data=[]
    image_list=[]
    landmark_list=[]
    attribute_list=[]
    feature_list=[]

    #打开数据集文件
    with open(TRAIN_FIEL, 'r') as f:
        #读取DATA_SIZE行的数据（每行数据就是一项数据记录）
        for i in range(DATA_SIZE):
            line_data = f.readline()
            if not line_data:
                break
                pass
            #将每行数据内容（str类型）用空格分隔成列表
            line_data = line_data.split()
            data.append(line_data)

            #获取图像rgb值矩阵，并加入image_list列表
            image_tensor=tf.image.decode_jpeg(tf.read_file("face_data_set/MTFL/"+line_data[0]),channels=1)
            image_list.append(tf.Session().run(image_tensor))

            #获取图像特征标签列表
            temp=[]
            for j in range(1,15):
                temp.append(float(line_data[j]))
            feature_list.append(temp)
            pass

    return image_list, feature_list  #返回图像列表和特征标签列表



if __name__ =='__main__':

    print("-----开始载入数据-----")
    x_list, y_list = input_data()
    print("-----已载入数据-----")
    x_valid, y_valid = x_list[:VALIDAION_SZIE], y_list[:VALIDAION_SZIE]
    x_train, y_train = x_list[VALIDAION_SZIE:], y_list[VALIDAION_SZIE:]
    
    prediction, error = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

    TRAIN_SIZE = len(x_train)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,SAVE_PATH)      #此行代码用于训练已有模型
    
    current_epoch = 0
    print("开始训练：")
    for i in range(EPOCHS):
        print("-----第",i+1,"次训练-----")
        for j in range(0,TRAIN_SIZE, BATCH_SIZE):
            sess.run(train_step, feed_dict={x:x_train[j:j+BATCH_SIZE],
                                            y_:y_train[j:j+BATCH_SIZE],
                                            keep_prob:0.5})
        #验证集验证模型并获得误差
        validation_loss = sess.run(error, feed_dict={x:x_valid, y_:y_valid, keep_prob: 1.0})
        print(validation_loss)
            
        #当此模型的误差小于阈值且小于前一个模型的误差，保存此模型
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            saver.save(sess,SAVE_PATH)
            print("保存--------")
        if current_epoch > EPOCHS:
            break
    sess.close()

