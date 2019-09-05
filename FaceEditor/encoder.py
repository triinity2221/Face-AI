import sys, random
import numpy as np
from matplotlib import pyplot as plt
from dutil import *



SHIFT_AMOUNT = 9
BATCH_SIZE = 8
NUM_KERNELS = 20
CONTINUE_TRAIN = True

NUM_EPOCHS = 2000
PARAM_SIZE = 80
LR = 0.001
NUM_RAND_FACES = BATCH_SIZE
NUM_TEST_FACES = BATCH_SIZE

def plotScores(scores, test_scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.plot(test_scores)
	plt.xlabel('Epoch')
	plt.ylim([0.0, 0.01])
	loc = ('upper right' if on_top else 'lower right')
	plt.legend(['Train', 'Test'], loc=loc)
	plt.draw()
	plt.savefig(fname)

#Load data set
print("Loading Data...")
f_image = r'D:/Compressed/FaceEditor-master/imgs/y_data1.npy'
y_train = np.memmap(f_image, dtype=np.uint8, mode='r', shape=(189280, 3, 40, 40))/255.0
print(y_train.shape)
y_train = y_train[:y_train.shape[0] - y_train.shape[0] % BATCH_SIZE]
x_train = np.expand_dims(np.arange(y_train.shape[0]), axis=1)
num_samples = y_train.shape[0]
print("Loaded " + str(num_samples) + " Samples.")

###################################
#  Create Model
###################################
print("Loading Keras...")
import os, math
# os.environ['THEANORC'] = "./gpu.theanorc"
# os.environ['KERAS_BACKEND'] = "theano"
# import theano
# print("Theano Version: " + theano.__version__)


from keras.initializers import RandomUniform
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

if CONTINUE_TRAIN:
	print("Loading Model...")
	model = load_model('Encoder.h5')
else:
	print("Building Model...")
	model = Sequential()

	model.add(Embedding(num_samples, PARAM_SIZE, input_length=1,name='Enbedder'))
	model.add(Flatten(name='pre_encoder'))
	print(model.output_shape)
	assert(model.output_shape == (None, PARAM_SIZE))
	
	model.add(Reshape((PARAM_SIZE, 1, 1), name='encoder'))
	print(model.output_shape)
	
	model.add(Conv2DTranspose(256, 4))           #(4, 4)
	model.add(Activation("relu"))
	print(model.output_shape)

	model.add(Conv2DTranspose(256, 4))                #(7, 7)
	model.add(Activation("relu"))
	print(model.output_shape)
	
	model.add(Conv2DTranspose(256, 4))                #(10, 10)
	model.add(Activation("relu"))
	print(model.output_shape)
	
	model.add(Conv2DTranspose(256, 4))     			#(13, 13)
	model.add(Activation("relu"))
	print(model.output_shape)
	
	model.add(Conv2DTranspose(128, 4, strides=2))     #(28, 28)
	model.add(Activation("relu"))
	print(model.output_shape)
	# oh = (input.h-1)*stride + kernel.h
	model.add(Conv2DTranspose(128, 4))     #(31, 31)
	model.add(Activation("relu"))
	print(model.output_shape)

	model.add(Conv2DTranspose(3, 10))      #(192, 144)
	model.add(Activation("sigmoid"))
	print(model.output_shape)
	# assert(model.output_shape[1:] == (3, 192, 144))

	model.compile(optimizer=Adam(lr=LR), loss='mse')
	plot_model(model, to_file='model.png', show_shapes=True)

###################################
#  Encoder / Decoder
###################################
print("Compiling SubModels...")
func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc_model = Model(inputs=model.input,
                  outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_FACES, PARAM_SIZE))

def make_rand_faces(rand_vecs, iters):
	x_enc = enc_model.predict(x_train, batch_size=BATCH_SIZE)
	
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	e, v = np.linalg.eig(x_cov)

	np.save('means.npy', x_mean)
	np.save('stds.npy', x_stds)
	np.save('evals.npy', e)
	np.save('evecs.npy', v)
	
	e_list = e.tolist()
	e_list.sort(reverse=True)
	plt.clf()
	plt.bar(np.arange(e.shape[0]), e_list, align='center')
	plt.draw()
	plt.savefig('evals.png')
	
	x_vecs = x_mean + np.dot(v, (rand_vecs * e).T).T

	y_faces = func([x_vecs, 0])[0]
	for i in range(y_faces.shape[0]):
		save_image(y_faces[i], 'rand' + str(i) + '.png')
		if i < 5 and (iters % 10) == 0:
			if not os.path.exists('morph' + str(i)):
				os.makedirs('morph' + str(i))
			save_image(y_faces[i], 'morph' + str(i) + '/img' + str(iters) + '.png')

make_rand_faces(rand_vecs, 0)
			
###################################
#  Train
###################################
print("Training...")
train_loss = []

for iters in range(NUM_EPOCHS):
	history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
	print(model.get_layer('Embedder').input)

	loss = history.history['loss'][-1]
	train_loss.append(loss)
	print("Loss: " + str(loss))

	plotScores(train_loss, [], 'EncoderScores.png', True)
	
	# if iters % 20 == 0:
	model.save('Encoder.h5')
	
	y_faces = model.predict(x_train[:NUM_TEST_FACES], batch_size=BATCH_SIZE)
	for i in range(y_faces.shape[0]):
		save_image(y_faces[i], 'gt' + str(i) + '.png')
	
	make_rand_faces(rand_vecs, iters)
	
	print("Saved")

print ("Done")
