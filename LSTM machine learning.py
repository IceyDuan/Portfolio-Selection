import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 股票数据导入及标准化
df1 = pd.read_excel('machine learning data 0.xlsx')
df2 = pd.read_excel('machine learning data 3.xlsx')
df3 = pd.read_excel('Example Returns 1.xlsx')

x_train = df1.values[0:100,1:268]
for i in range(x_train.shape[1]):
    mean1 = np.mean(x_train[:,[i]])
    std1 = np.std(x_train[:,[i]])
    x_train[:,[i]] = (x_train[:,[i]]-mean1)/std1
y_train = df1.values[0:100,268]

x_test = df2.values[0:50,1:268]
for j in range(x_test.shape[1]):
    mean2 = np.mean(x_test[:,[j]])
    std2 = np.std(x_test[:,[j]])
    x_test[:,[j]] = (x_test[:,[j]]-mean2)/std2
y_test = df2.values[0:50,268]

x_example = df3.values[0:15,1:268]
for k in range(x_test.shape[1]):
    mean3 = np.mean(x_example[:,[k]])
    std3 = np.std(x_example[:,[k]])
    x_example[:,[k]] = (x_example[:,[k]]-mean3)/std3

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')
x_example = x_example.astype('float32')
y_test = y_test.astype('int')
y_train = y_train.astype('int')
x_train,x_test,x_example = tf.reshape(x_train,[len(x_train),267,-1]),tf.reshape(x_test,[len(x_test),267,-1]),tf.reshape(x_example,[len(x_example),267,-1])

# 设计网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=40,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.LSTM(units=25,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(units=15, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(units=3, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.001))
], name="LSTM")

# 定义代价函数
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 模型编译
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=loss_function, metrics=['accuracy'])
# 模型训练及应用
history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=50,
                    validation_data=(x_test, y_test))
model.summary()
model.save('Stock Prediction Model.keras')
logits1 = model(x_test)
logits2 = model(x_train)
logits3 = model(x_example)
y_test_predict_prob = tf.nn.softmax(logits1)
y_train_predict_prob = tf.nn.softmax(logits2)
y_example_predict_prob = tf.nn.softmax(logits3)
y_test_predict = np.argmax(y_test_predict_prob,axis=1)
y_train_predict = np.argmax(y_train_predict_prob,axis=1)
y_example_predict = np.argmax(y_example_predict_prob,axis=1)
count1 = sum(1 for x, y in zip(y_test_predict,y_test) if x == y)
count2 = sum(1 for x, y in zip(y_train_predict,y_train) if x == y)
print('Accuracy for test set:',count1/50)
print('Accuracy for train set:',count2/100)

# 结果导出
y_train_predict = pd.DataFrame(y_train_predict)
y_test_predict = pd.DataFrame(y_test_predict)
y_example_predict = pd.DataFrame(y_example_predict)
y_train_predict.to_excel('Result train.xlsx', index=True)
y_test_predict.to_excel('Result test.xlsx', index=True)
y_example_predict.to_excel('Result example.xlsx', index=True)

# 定义绘图函数
def print_history(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss Curve')
    plt.xlabel('Epoch')
    plt.legend(['Loss'])
    plt.show()

print_history(history)
tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)