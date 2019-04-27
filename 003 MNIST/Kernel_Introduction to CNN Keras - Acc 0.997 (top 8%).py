#1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
#Dense表示这个神经层是全连接层
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#优化器采用RMSprop，加速神经网络训练方法
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
sns.set(style='white', context='notebook', palette='deep')

#2
f1=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C3_手写数字识别/train.csv")
f2=open("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C3_手写数字识别/test.csv")
train = pd.read_csv(f1)
test = pd.read_csv(f2)

#3
#要把X和预测的y分开一下
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
del train #用这种方式节省空间
g = sns.countplot(Y_train)#查看每种数字的数量
Y_train.value_counts()

#4
#检查是否有缺失数据
print(X_train.isnull().any().describe(),'\n',test.isnull().any().describe())

#5
#标准化
#利用grayscale normalization来避免光影的影响
#CNN需要将0-255的值收敛成0-1
X_train = X_train / 255.0
test = test / 255.0

#6
#将二维数据变为三维
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#特征因子化
#to_categorical是Keras中的onehot编码方法
Y_train = to_categorical(Y_train, num_classes = 10)

#7
#分离训练集和交叉验证集
random_seed = 2
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=random_seed)

#8
#实际观测一下（imshow本来是绘制热图的，这里用也真合适）
g = plt.imshow(X_train[0][:,:,0])

#9
# Set the CNN model
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
'''
Keras搭建神经网络的步骤：
Step1：选择模型
Step2：构建网络层
Step3：编译
Step4：训练
Step5：预测
'''
'''Step1：选择模型（序贯模型或函数式模型）这里是序贯模型'''
model = Sequential()#Sequential建立模型
'''Step2：构建网络层（输入层、隐藏层、输出层）'''
#Conv2D：实现卷积的函数
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#Dropout：一定的概率暂时丢弃某个单元网格，防止过拟合
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#10
# 定义优化器RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
'''Step3：编译（优化器、损失函数、性能评估）'''
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#11
#学习率退火器
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

#12
# Without data augmentation i obtained an accuracy of 0.98114
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
#          validation_data = (X_val, Y_val), verbose = 2)

# With data augmentation to prevent overfitting (accuracy 0.99286)
#数据增强
datagen = ImageDataGenerator(
        featurewise_center=False,  # 输入数据集去中心化（均值为0）
        samplewise_center=False,  # 使输入数据样本均值为零
        featurewise_std_normalization=False,  # 标准化：数据集除以标准差
        samplewise_std_normalization=False,  # 输入的每个样本除以自身的标准差
        zca_whitening=False,  #对输入数据施加ZCA白化
        rotation_range=10,  # 数据提升时图片随机旋转的角度（0-180）
        zoom_range = 0.1, #
        width_shift_range=0.1,  # 数据提升时图片随机水平偏移的幅度（图片宽度的某个比例）
        height_shift_range=0.1,  # 数据提升时图片随机竖直偏移的幅度（图片高度度的某个比例）
        horizontal_flip=False,  # 随机对图片水平翻转
        vertical_flip=False)  # 随机进行竖直翻转
datagen.fit(X_train)

#13
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
'''
参数解释：
batch_size：训练多少个参数更新一下权重
epochs：训练的轮数
validation_data：验证集
steps_per_epoch：将一个epoch分为多少个steps，也就是划分一个batch_size多大
verbose：0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
callbacks：训练中会调用的回调函数
'''

#14
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

#15
#查看混合矩阵（误差矩阵）

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''Step5：预测'''
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

#16
# 查看测试集中预测错误的样本
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

#17
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

#18
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("E:/Tinky/大学课件及作业/6 自学课/6-3.Kaggle竞赛/C3_手写数字识别/submission_Exp.csv",index=False)