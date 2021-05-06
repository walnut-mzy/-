# -
这是笔者做的一个验证码识别的项目

# 验证码识别

## 数据收集

这里数据收集我们使用了河南某211高校的验证码数据集：

收集一万张图片做数据集：

```python
import requests

for i in range(1,10000):
    response=requests.get(url="https://jksb.v.zzu.edu.cn/vls6sss/zzjlogin3d.dll/zzjgetimg?ids="+str(i))
    print(i)
    with open("photos/"+str(i)+".jpg","wb") as fp:
        fp.write(response.content)
```

收集的验证码如图：

![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-D7f0NP1E-1620293521986)(C:\Users\mzy\AppData\Roaming\Typora\typora-user-images\image-20210506171232247.png)\]](https://img-blog.csdnimg.cn/20210506173319415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


## 图片处理

### 图片二值化处理

```python
#图片二值化
threshold = 15

table = []
for i in range(256):
    if i < threshold:
        table.append(1)
    else:
        table.append(0)

img = img.point(table, '1')
plt.imshow(img)
```

![](https://img-blog.csdnimg.cn/20210506173335225.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


### 去除噪点

```python
def zaodian(binImg):
    a=0
    while a<=20:
        a+=1
        pixdata = binImg.load()
        width, height = binImg.size
        for y in range(1, height- 1):
            for x in range(1, width- 1):
                count = 0
                sudi=0
                if pixdata[x, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x, y + 1] == sudi:
                    count = count + 1
                if pixdata[x - 1, y] == sudi:
                    count = count + 1
                if pixdata[x + 1, y] == sudi:
                    count = count + 1
                if pixdata[x - 1, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x - 1, y + 1] == sudi:
                    count = count + 1
                if pixdata[x + 1, y - 1]== sudi:
                    count = count + 1
                if pixdata[x + 1, y + 1] == sudi:
                    count = count + 1
                if count >6:
                    pixdata[x, y] =0
            #将边框的值全部设为0
            for i in range(1, height):

                pixdata[1, i] = 0
                pixdata[width-1,i]=0
            for j in range(1, width):

                pixdata[j, 1] = 0
                pixdata[j,height-1]=0
    return binImg
```

#这里噪点的去除用了种子染色法：具体过程是循环图片上每一个点，判读图片每一个点周围的像素值如果小于如果小于某个值，cout数减一，如果cout数大于某一个值则将这个噪点去除#

![](https://img-blog.csdnimg.cn/20210506173404743.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


### 文字分割

```python
def segment_image(image):
    labeled_image = label(image > 0)  # 找出显示的连接在一起的像素块
    subimages = []  # 分割的小图像集合
    for region in regionprops(labeled_image):  # regionprops统计被标记的区域的面积分布，显示区域总数
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])  # 分割图像
    if len(subimages) == 0:
        return [image, ]
    print(subimages)
    return subimages
```

![](https://img-blog.csdnimg.cn/20210506173419519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


可以看出分割的图片并不是那么规则，在训练过程的时候，并不能保证输入形状一致，这里我们需要对图片进行放缩处理，（这里笔者采用过resize函数但是由于其广播机制，处理后的图片并不是那么美观于是这里我们使用了padding函数进行填充）

#### pad函数的应用

[numpy中pad函数的常用方法 - AI菌 - 博客园 (cnblogs.com)](https://www.cnblogs.com/hezhiyao/p/8177541.html)

应用将一个数组转换为28*28的形式

```python
    for i in k:
        print("hello")
        plt.imshow(i)
        plt.show()
        shape=i.shape
        shape_x=shape[0]
        shape_y=shape[1]
        # print(shape_x,shape_y)
        # print(((int((28-shape_x)/2),int((28-shape_y)/2)),((28-shape_y)-int((28-shape_y)/2),(28-shape_x)-int((28-shape_x)/2))))
        c=np.pad(i,pad_width=((int((28-shape_x)/2),(28-shape_x)-int((28-shape_x)/2)),(int((28-shape_y)/2),(28-shape_y)-int((28-shape_y)/2))),mode='constant',constant_values=(0,0))
        plt.imshow(c)
        plt.show()
        print(c.shape)

```

![](https://img-blog.csdnimg.cn/20210506173432203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


这样处理就可以使图片转化为指定形状大小。

## 训练集的训练与构建

### 训练集的构建

这里我们把图片数据和其标签保存成csv格式：

```python
  print("imge show")
            print(c.shape)
            data1=c.flatten()
            data1=pd.DataFrame(data1.reshape((1,len(data1))))
            chr_4=""
            while len(chr_4)!=1:
                chr_4=input()
                print("input error")
            strs = ""
            for arr in c:
                strs = strs + ','.join(str(i) for i in arr) + ","
            # print(strs,len(strs))
            str_1 = str_1 + chr_4 + "," + strs + "\n"
        with open("train_test_letter_exam_a.csv", "a") as f:
            f.write(str_1)
            f.close()
  
```

总和代码：

```python
from skimage.measure import label, regionprops
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def image_deals(train_file):       # 读取原始文件

    image_resized = tf.image.resize(train_file, [28, 28]) / 255.0   #把图片转换为255*255的大小
    #print(image_resized,label)
    return image_resized
def zaodian(binImg):
    a=0
    while a<=20:
        a+=1
        pixdata = binImg.load()
        width, height = binImg.size
        for y in range(1, height- 1):
            for x in range(1, width- 1):
                count = 0
                sudi=0
                if pixdata[x, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x, y + 1] == sudi:
                    count = count + 1
                if pixdata[x - 1, y] == sudi:
                    count = count + 1
                if pixdata[x + 1, y] == sudi:
                    count = count + 1
                if pixdata[x - 1, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x - 1, y + 1] == sudi:
                    count = count + 1
                if pixdata[x + 1, y - 1]== sudi:
                    count = count + 1
                if pixdata[x + 1, y + 1] == sudi:
                    count = count + 1
                if count >6:
                    pixdata[x, y] =0
            for i in range(1, height):

                pixdata[1, i] = 0
                pixdata[width-1,i]=0
            for j in range(1, width):

                pixdata[j, 1] = 0
                pixdata[j,height-1]=0
    return binImg

def segment_image(image):
    labeled_image = label(image > 0)  # 找出显示的连接在一起的像素块
    subimages = []  # 分割的小图像集合
    for region in regionprops(labeled_image):  # regionprops统计被标记的区域的面积分布，显示区域总数
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])  # 分割图像
    if len(subimages) == 0:
        return [image, ]
    print(subimages)
    return subimages
for i in range(1,1346):
    img = Image.open("photos/"+str(i)+".jpg")
    #plt.imshow(img)
    #plt.show()

    #图片二值化
    threshold = 15

    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)

    img = img.point(table, '1')
    plt.imshow(img)
    #plt.show()

    #去除噪点
    img_1=zaodian(img)
    #plt.imshow(img_1)
    #plt.show()

    # #灰度化
    # img_L = img_1.convert("L")
    # plt.imshow(img_L)
    # plt.show()

    #nrImg.show()
    c_1=np.array(img_1)
    k = segment_image(c_1)
    print(len(k))
    if len(k) == 4:
        str_1=""
        for i in k:
            print("hello")
            #plt.imshow(i)
            #plt.show()
            shape=i.shape
            shape_x=shape[0]
            shape_y=shape[1]
            # print(shape_x,shape_y)
            # print(((int((28-shape_x)/2),int((28-shape_y)/2)),((28-shape_y)-int((28-shape_y)/2),(28-shape_x)-int((28-shape_x)/2))))
            c=np.pad(i,pad_width=((int((30-shape_x)/2),(30-shape_x)-int((30-shape_x)/2)),(int((30-shape_y)/2),(30-shape_y)-int((30-shape_y)/2))),mode='constant',constant_values=(0,0))
            plt.imshow(c)
            plt.show()
            print("imge show")
            print(c.shape)
            data1=c.flatten()
            data1=pd.DataFrame(data1.reshape((1,len(data1))))
            chr_4=""
            while len(chr_4)!=1:
                chr_4=input()
                print("input error")
            strs = ""
            for arr in c:
                strs = strs + ','.join(str(i) for i in arr) + ","
            # print(strs,len(strs))
            str_1 = str_1 + chr_4 + "," + strs + "\n"
        with open("train_test_letter_exam_a.csv", "a") as f:
            f.write(str_1)
            f.close()
    else:
        print("error")

```



### 训练集的训练
```python
import tensorflow as tf
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("train_test_letter_exam_a.csv")
print(data)
# #将csv数据打印下来成列表形式
# print(data.head())
list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
     "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
     "w", "x", "y", "z"
     ]
x=data.iloc[:,1:]/255
print(x)
y_1=data.iloc[:,:1]
#print(y)
for i in range(len(y_1)):
    #print(i)
    y_1.iloc[i,0]=list.index(y_1.iloc[i,0])
    print(y_1.iloc[i,0])
y=y_1.astype('int32')
# print(y)
# print(x)
#y_onehot=tf.keras.utils.to_categorical(str(y))
#print(data,x,y)
# #data.columns获取数据列数，data获取数据行数
# print(len(data),len(data.columns))
#print(data,len(data.nrows))
#data.iloc[:,-1].value_counts()  统计最后一行的每个数据个数

#[tf.keras.layers.Dense(100,input_shape=(3,))] 输入层
#这里的100代表着中间神经元的个数 input_shape=代表输入数据的形状，activation是激活函数
print("++++",len(x.columns))
model=tf.keras.Sequential(
    [
        tf.keras.layers.Dense(3200, input_shape=(len(x.columns),), activation="relu"),
        tf.keras.layers.Dense(1600,activation="relu"),
        #随机丢弃70%的值
        tf.keras.layers.Dense(1600,activation="relu"),
        # tf.keras.layers.Dropout(0.7),
        # tf.keras.layers.Dense(100,activation="sigmoid"),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(36,activation="softmax"),
     ]
                          )
#显示模型参数
model.summary()
##train models error 设置算法为梯度下降算法 损失函数为均方差
# model.compile(optimizer='adam',loss='mse'
#               )
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy']
              )  #metrics=["acc"]度量正确率
#train models
model.fit(x,y,epochs=80,batch_size=32)
#train
test=pd.read_csv("train_test_letter_exam_a.csv",error_bad_lines=False)
#print(test)
test_x=test.iloc[:,1:]/255
print(test_x)
print(test_x)
test_y=test.iloc[:,:1]
#test_y_onehot=tf.keras.utils.to_categorical(test_y)
print(test.iloc[0,5])
#print(test_y)
predict=model.predict(test_x)
l=[]
for i in range(len(test_x)):
    if list[int(numpy.argmax(predict[i]))]==test_y.iloc[i,0]:
        l.append(1)
    else:
        l.append(0)
    print("train test:",list[int(numpy.argmax(predict[i]))],"   ","really test:",test_y.iloc[i,0])
#calculate the performance score ,the fraction of correct answers
scorecard_array=numpy.asarray(l)
print("performance=",scorecard_array.sum()/scorecard_array.size)
model.save("Model/neuralNetwork1_tensorflow.h5")
with open("train_result.txt","w") as fp:
    fp.write("performance="+str(scorecard_array.sum()/scorecard_array.size))

#print(test_y,"\n",predict[0])
```
