import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten
from keras.callbacks import TensorBoard

train_data = pd.read_csv("../csv_files/fer2013LimitedImages.csv")
train = train_data[["emotion", "pixels"]][train_data["Usage"] == "Training"]
train['pixels'] = train['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
train_pix = np.vstack(train['pixels'].values)
test = train_data[["emotion", "pixels"]][train_data["Usage"] == "PublicTest"]
test['pixels'] = test['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
test_pix = np.vstack(test['pixels'].values)

train_ind = train["emotion"].values
test_ind = test["emotion"].values

train_pix = train_pix.reshape(-1, 48, 48, 1)
train_ind = np_utils.to_categorical(train_ind)
test_pix = test_pix.reshape(-1, 48, 48, 1)
test_ind = np_utils.to_categorical(test_ind)

model = Sequential()
model.add(Conv2D(64, 5, data_format="channels_last", kernel_initializer="he_normal",
                 input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 5))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Conv2D(64, 5))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 5))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

logger = TensorBoard(log_dir='logs', write_graph=True, histogram_freq=5)
model.summary()

model.fit(train_pix,
          train_ind,
          epochs=10,
          shuffle=True,
          batch_size=100,
          verbose=2,
          validation_data=(test_pix, test_ind),
          callbacks=[logger])

model.save("../model/trainModel.h5")

model_json = model.to_json()
with open("../model/trainedModel.json", 'w') as json_file:
    json_file.write(model_json)
