# import 
import tensorflow as tf 
from tensorflow.keras.layers import LSTM,Dense,Dropout,CuDNNLSTM
from tensorflow.keras import Sequential
import numpy as np 


# getting datasets
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()

train_x = train_x/255.0
test_x = test_x/255.0

# shape Configurations

print(train_x.shape)
print(train_x[0].shape)
print(train_y.shape)
print(test_x.shape)
print(test_x[0].shape)
print(test_y.shape)


# saving check point for model
# checkpoint_path = "mymodels/mnist_keras_rnn.ckpt"
# checkpoint_path = "mymodels/mnist_keras_rnn-{epochs:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_weights_only=True)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_weights_only=True,period=5)



# Building Model

model = Sequential()
model.add(CuDNNLSTM(128,input_shape=(train_x[0].shape),return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

# compiling the model

op = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',optimizer=op,metrics=['accuracy'])


# Training the model

history = model.fit(train_x,train_y,epochs=3,validation_data=(test_x,test_y))
# model.save_weights(checkpoint_path.format(epochs=0))
# model.save_weights("mymodels/mnist_keras_rnn")
# history = model.fit(train_x,train_y,epochs=3,validation_data=(test_x,test_y),callbacks=[cp_callback])
# model = create_model()
# model.load_weights(checkpoint_path)
# model.summary()


# model Summary
model.summary()
model.save("mymodels/mnist_keras_rnn.h5")

# Testing

test_loss ,test_acc = model.evaluate(test_x,test_y)

print(test_loss,test_acc) 


# model = tf.keras.models.load_model("mymodels/mnist_keras_rnn.h5")
# model.summary()