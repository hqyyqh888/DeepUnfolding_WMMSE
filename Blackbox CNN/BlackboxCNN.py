
# Supervised + Unsupervised Learning using tensorflow

import os  
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 只显示 warning 和 Error   
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error 

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

def generate_index_matrix(X_index):
    X_index = X_index
    index_list = list()
    for i in range(X_index.shape[0]):
        index = [1] * (12 * User)
        for k in range(User):
            if X_index[i][k] == 0:
                index[4 * k + 3] = 0
                index[4 * k + 1] = 0
                index[4 * k + 3 + 4 * User] = 0
                index[4 * k + 1 + 4 * User] = 0
                index[8 * User + 3 * k + 1] = 0
                index[8 * User + 3 * k + 2] = 0
                index[8 * User + 3 * User + k] = 0
        index_list.append(index)

    index_matrix = np.array(index_list)
    return index_matrix


def merge_index(inputs):
    x, index = inputs
    return x * index


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积核数目
            kernel_size=[3, 3],  # 感受野大小
            input_shape=(Nr * User, Nr * User, 1),
            padding="same",  # padding策略
            kernel_initializer='uniform'
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积核数目
            kernel_size=[3, 3],  # 感受野大小
            padding="same",  # padding策略
            kernel_initializer='uniform'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积核数目
            kernel_size=[3, 3],  # 感受野大小
            padding="same",  # padding策略
            kernel_initializer='uniform'
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.leaky_relu3 = tf.keras.layers.LeakyReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense0 = tf.keras.layers.Dense(units=1024)
        self.relu0 = tf.keras.layers.ReLU()
        self.dropout0 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(units=12 * User)

        self.dense2 = tf.keras.layers.Dense(units=64, input_shape=(User,))
        self.relu2 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=256)
        self.relu3 = tf.keras.layers.ReLU()
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense4 = tf.keras.layers.Dense(units=12 * User)

        self.lambda1 = tf.keras.layers.Lambda(merge_index)

    def call(self, inputs):
        X_matrix = inputs[:, :Nr * User * Nr * User * 1]
        X_index = inputs[:, -User:]
        X_index = tf.cast(X_index, tf.float32)

        X_matrix = tf.reshape(X_matrix, (-1, 1, Nr * User, Nr * User))
        X_matrix = tf.transpose(X_matrix, perm=[0, 2, 3, 1])
        X_matrix = tf.cast(X_matrix, tf.float32)

        x = self.conv1(X_matrix)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.leaky_relu2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.leaky_relu3(x)
        x = self.flatten(x)
        # x = self.dense0(x)
        # x = self.relu0(x)
        # x = self.dropout0(x)
        x = self.dense1(x)

        x2 = self.dense2(X_index)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        # x2 = self.dense3(x2)
        # x2 = self.relu3(x2)
        # x2 = self.dropout3(x2)
        x2 = self.dense4(x2)
        #print(x2)

        prediction = self.lambda1([x, x2])

        return prediction

    def predict(self, inputs):
        logits = self(inputs)
        return logits


def get_unsupervised_loss(y_logit_pred, X):
    size = X.numpy().shape[0]

    Weight = X[:, -2 * User:-User]
    Weight = tf.cast(Weight, tf.float32)

    index = X[:, -User:]
    index = generate_index_matrix(index.numpy())

    X = X[:, Nr * User * Nr * User:-2 * User]
    X = tf.reshape(X, (-1, 2, Nr * User, Nt))
    X = tf.transpose(X, perm=[0, 2, 3, 1])
    X = tf.complex(X[:, :, :, 0], X[:, :, :, 1], name=None)
    X = tf.cast(X, tf.complex64)
    new_H = tf.reshape(X, (-1, User, Nr, Nt))

    y_logit_pred = y_logit_pred * index

    UU = y_logit_pred[:, :8 * User]
    UU = tf.reshape(UU, (-1, 2, User, 2, 2))
    UU = tf.transpose(UU, perm=[0, 2, 3, 4, 1])
    UU = tf.complex(UU[:, :, :, :, 0], UU[:, :, :, :, 1], name=None)
    UU = tf.cast(UU, tf.complex64)

    WW = y_logit_pred[:, 8 * User:]
    real_WW = tf.cast(WW[:, :3 * User], tf.float32)
    imag_WW = tf.cast(WW[:, 3 * User:], tf.float32)

    loss = 0
    for i in range(size):
        obj = 0
        trace_V = 0
        HH = X[i]
        alpha = Weight[i]
        H = new_H[i]
        U = UU[i]

        # construct W
        real_W = real_WW[i]
        imag_W = imag_WW[i]
        temp_W = list()
        for k in range(User):
            real_Wk = real_W[3 * k:3 * k + 3]
            imag_Wk = imag_W[k]
            Wk = tf.complex([[real_Wk[0], real_Wk[1]], [real_Wk[1], real_Wk[2]]], [[0, imag_Wk], [-imag_Wk, 0]],
                            name=None)
            temp_W.append(Wk)
        W = tf.stack(temp_W, 0)

        HHT = tf.matmul(HH, tf.transpose(HH, conjugate=True))

        temp = tf.zeros([2 * User, 2 * User], tf.complex64)
        for k in range(User):
            HHU = tf.matmul(HHT[:, Nr * k:Nr * (k + 1)], U[k])
            trace_UWU = tf.trace(tf.matmul(tf.matmul(U[k], W[k]), tf.transpose(U[k], conjugate=True)))
            temp += tf.matmul(tf.matmul(HHU, W[k]), tf.transpose(HHU, conjugate=True)) + trace_UWU * HHT

        try:
            temp_inverse = tf.matrix_inverse(temp, adjoint=None, name=None)
        except:
            temp += tf.ones([2 * User, 2 * User], tf.complex64) * 1e-4
            temp_inverse = tf.matrix_inverse(temp, adjoint=None, name=None)

        # calculate X and V
        temp_V = list()
        for k in range(User):
            Xk = tf.matmul(temp_inverse, tf.matmul(tf.matmul(HHT[:, Nr * k:Nr * (k + 1)], U[k]), W[k]))
            Vk = tf.matmul(tf.transpose(HH, conjugate=True), Xk)
            temp_V.append(Vk)
        V = tf.stack(temp_V, 0)
        VV = tf.zeros([Nt, Nt], tf.complex64)

        # calculate obj
        VkVk_list = list()
        for k in range(User):
            VkVk = tf.cast(tf.matmul(V[k], tf.transpose(V[k], conjugate=True)), tf.complex64)
            VkVk_list.append(VkVk)
            VV += VkVk
            trace_V += tf.trace(VkVk)
        for k in range(User):
            Hk = H[k]
            VkVk = VkVk_list[k]
            a = tf.matmul(Hk, (VV - VkVk))
            Jk = trace_V * tf.cast(tf.eye(Nr), tf.complex64) + tf.matmul(a, tf.transpose(Hk, conjugate=True))
            HVk = tf.matmul(Hk, V[k])
            HVVH = tf.matmul(HVk, tf.transpose(HVk, conjugate=True))
            b = tf.matmul(HVVH, tf.matrix_inverse(Jk))
            obj += alpha[k] * tf.real(tf.log(tf.matrix_determinant(tf.cast(tf.eye(Nr), tf.complex64) + b)))
        obj = -obj
        loss += obj
    #print(loss / size)
    return loss / size


User = 30  #The number of users
Nt = 64  #The number of transmit antennas
Nr = 2 #The number of receive antennas
batch_size = 512
learning_rate = 0.001
num_epoch = 100

model = CNN()

input_data = pd.read_csv('DataSet\Input_H.csv', sep=",", header=None).values
output_data = pd.read_csv('DataSet\Output_UW.csv', sep=",", header=None).values

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.05)

data_size = X_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(data_size)
dataset = dataset.batch(batch_size)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for epoch in range(num_epoch):
    for X, y in dataset:
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X))
            supervised_loss = tf.losses.huber_loss(y, y_logit_pred)
            loss = supervised_loss
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print("epoch %d: loss %f" % (epoch, loss.numpy()))

X_test_index = X_test[:, -User:]
X_test_index_matrix = generate_index_matrix(X_test_index)

y_pred = model.predict(tf.constant(X_test)).numpy()
y_pred = y_pred * X_test_index_matrix
data_predict = pd.DataFrame(y_pred)
data_predict.to_csv('DataSet\Predict_UW_sup.csv', header=None, index=None)

data_test = pd.DataFrame(X_test)
data_test.to_csv('DataSet\Test_H_unsup.csv', header=None, index=None)

data_true = pd.DataFrame(y_test)
data_true.to_csv('DataSet\True_UW_unsup.csv', header=None, index=None)

model.dense2.trainable = False
model.dense3.trainable = False
model.dense4.trainable = False

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(data_size)
dataset = dataset.batch(batch_size)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate / 10)
for epoch in range(20):
    for X, y in dataset:
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X))
            unsupervised_loss = get_unsupervised_loss(y_logit_pred, tf.convert_to_tensor(X))
            loss = unsupervised_loss
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print("epoch %d: loss %f" % (epoch, loss.numpy()))
    y_pred = model.predict(tf.constant(X_test)).numpy()
    y_pred = y_pred * X_test_index_matrix

    data_predict = pd.DataFrame(y_pred)
    data_predict.to_csv('DataSet\Predict_UW_unsup.csv', header=None, index=None)
