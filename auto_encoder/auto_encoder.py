from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

batch_size = 32
epochs = 10
saveLogDir = './log'
saveDir = './model/'

def encoder(input_img):
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	return encoded

def decoder(input_img, encoded):
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
	decoded = Activation('sigmoid')(x)

	model = Model(input_img, decoded)

	return model

def auto_encoder():
	input_img = Input(shape=(32, 32, 3))
	encoded = encoder(input_img)
	auto_encoded = decoder(input_img, encoded)
	return auto_encoded

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_val = x_test[:9000]
x_test = x_test[9000:]

model = auto_encoder()
model.compile(optimizer='adam', loss='binary_crossentropy')
# model.summary()

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
tb_cb = TensorBoard(log_dir=saveLogDir, histogram_freq=1, write_graph=True)
cp_cb = ModelCheckpoint(filepath=saveDir+'AE_Cifar10.{epoch:02d}.hdf5', \
	monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


history = model.fit(x_train, x_train,
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	validation_data=(x_val, x_val),
	callbacks=[es_cb, tb_cb, cp_cb],
	shuffle=True
	)


c10test = model.predict(x_test)

plt.figure(figsize=(10, 4))
n = 6
for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i])
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(c10test[i])
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
