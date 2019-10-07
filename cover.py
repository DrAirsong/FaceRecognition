import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse

#python文件参数设置
parse = argparse.ArgumentParser()
parse.add_argument('--src_img',default='01.jpg')
parse.add_argument('--mask_img',default='doge01.png')

args = parse.parse_args()

SAVE_PATH = 'model' #训练好的模型路径
#IMAGE_PATH = args.src_img   #输入人脸图像
#MASK_PATH = args.mask_img   #输入表情包图像

IMAGE_PATH = '01.jpg'
MASK_PATH = 'doge01.png'

IMAGE_SIZE = 250    #人脸图像大小

#—————————————以下是与face_recognition.py文件中相同的卷积神经网络模型—————————————
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None,250,250,1])/255.
y_ = tf.placeholder(tf.float32, shape=[None, 14]) 
keep_prob = tf.placeholder(tf.float32)

def model():
    w_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, w_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

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

    prediction = tf.matmul(h_fc2_drop, w_fc3)+b_fc3
    
    #cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(y_-prediction)))

    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_-prediction)))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(prediction),reduction_indices=[1]))

    return prediction, rmse
#—————————————以上是与face_recognition.py文件中相同的卷积神经网络模型—————————————

#将人脸图像放入模型，得到图像特征（包括所需的五个关键点坐标），以列表形式存储
def get_value_list():

    prediction, rmse = model()
    image_list=[]
    image = cv.imread(IMAGE_PATH,0)
    image = cv.resize(image,(250,250))
    image = np.reshape(image,(250,250,1))

    image_list.append(image)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, SAVE_PATH)
    value_list=sess.run(prediction, feed_dict={x:image_list,keep_prob:1})
    
    sess.close()

    return value_list

#处理表情包图像，将4通道转为3通道
#若某一像素点的alpha通道为0，则用热区图像对应点代替
def deal_mask(imageROI, mask):
    mat = np.empty([np.shape(mask)[0], np.shape(mask)[1], 3],np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j][3] == 0:
                mat[i][j] = imageROI[i][j]
            else:
                mat[i][j][0] = mask[i][j][0]*(mask[i][j][3]/255)
                mat[i][j][1] = mask[i][j][1]*(mask[i][j][3]/255)
                mat[i][j][2] = mask[i][j][2]*(mask[i][j][3]/255)

    return mat

if __name__=='__main__':
    print("-----嵌入表情包-----")

    #获得人脸图像并调整大小
    cap = cv.VideoCapture(IMAGE_PATH)
    has_frame, frame = cap.read()
    frame = cv.resize(frame, (IMAGE_SIZE,IMAGE_SIZE),interpolation=cv.INTER_AREA)

    #获得人脸图像的关键点的坐标列表
    value_list=get_value_list()
    value_list = value_list[0]

    key_points=[]
    rectangle_points=[]
   
    for i in range(5):
        x = value_list[i]
        y = value_list[i+5]
        point = (int(x), int(y))
        key_points.append(point)
        
    #根据五个关键点的坐标获得人脸矩形框的坐标，存储在列表里
    sub_width = (key_points[1][0]-key_points[0][0])/2
    sub_height = ((key_points[0][1]-key_points[2][1])+(key_points[1][1]-key_points[2][1]))/2/2*3
    rectangle_points.append((int(key_points[2][0]-sub_width/2*4), int(key_points[2][1]+sub_height*2)))
    rectangle_points.append((int(key_points[2][0]+sub_width/2*4), int(key_points[2][1]+sub_height*2)))
    rectangle_points.append((int(key_points[2][0]-sub_width/2*4), int(key_points[2][1]-sub_height*4/3)))
    rectangle_points.append((int(key_points[2][0]+sub_width/2*4), int(key_points[2][1]-sub_height*4/3)))

    rectangle_height = rectangle_points[3][1] - rectangle_points[0][1]
    rectangle_width = rectangle_points[1][0] - rectangle_points[0][0]

    #下列注释代码是用来画关键点及矩形框的
    #cv.ellipse(frame, key_points[0], (3, 3), 0, 0, 360, (0, 0, 255),-1)
    #cv.ellipse(frame, key_points[1], (3, 3), 0, 0, 360, (0, 0, 255), -1)
    #cv.ellipse(frame, key_points[2], (3, 3), 0, 0, 360, (255, 0, 0), -1)
    #cv.ellipse(frame, key_points[3], (3, 3), 0, 0, 360, (0, 255, 0),-1)
    #cv.ellipse(frame, key_points[4], (3, 3), 0, 0, 360, (0, 255, 0), -1)

    #cv.line(frame, rectangle_points[0], rectangle_points[1], (0, 255, 0), 3)
    #cv.line(frame, rectangle_points[1], rectangle_points[3], (0, 255, 0), 3)
    #cv.line(frame, rectangle_points[3], rectangle_points[2], (0, 255, 0), 3)
    #cv.line(frame, rectangle_points[2], rectangle_points[0], (0, 255, 0), 3)
    
    #读入及处理待嵌入的表情包图像
    mask = cv.imread(MASK_PATH,-1)
    mask_height = np.shape(mask)[0]
    mask_width = np.shape(mask)[1]
    h = int(mask_height/2)
    w = int(mask_width/2)

    imageROI = frame[key_points[2][1]-h:key_points[2][1]+h, 
                     key_points[2][0]-w:key_points[2][0]+w]

    #根据热区图像处理表情包
    mask = deal_mask(imageROI, mask)   

    mask = cv.resize(mask,(np.shape(imageROI)[1],np.shape(imageROI)[0]))
   
    #嵌入表情包
    frame[key_points[2][1]-h:key_points[2][1]+h, 
          key_points[2][0]-w:key_points[2][0]+w] = mask

    #显示图像
    cv.imshow("image", frame)

    cv.waitKey(0)
  



