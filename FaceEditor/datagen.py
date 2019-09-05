import os, random, sys
import numpy as np
import cv2
from dutil import *

NUM_IMAGES = 9464
SAMPLES_PER_IMG = 10
DOTS_PER_IMG = 60
IMAGE_W = 40
IMAGE_H = 40
IMAGE_DIR = 'D:\Compressed\FaceEditor-master\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'
NUM_SAMPLES = NUM_IMAGES * 2 * SAMPLES_PER_IMG

def center_resize(img):
	assert(IMAGE_W == IMAGE_H)
	w, h = img.shape[0], img.shape[1]
	if w > h:
		x = (w-h)/2
		img = img[x:x+h,:]
	elif h > w:
		img = img[:,0:w]
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)

def yb_resize(img):
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
	
def rand_dots(img, sample_ix):
	sample_ratio = float(sample_ix) / SAMPLES_PER_IMG
	return auto_canny(img, sample_ratio)

ix = 0
f1 = r'D:/Compressed/FaceEditor-master/imgs/x_data1.npy'
f2 = r'D:/Compressed/FaceEditor-master/imgs/y_data1.npy'
x_data = np.memmap(f1,mode='w+',shape=(NUM_SAMPLES, 1, IMAGE_H, IMAGE_W), dtype=np.uint8)
y_data = np.memmap(f2,mode='w+',shape=(NUM_SAMPLES, 3, IMAGE_H, IMAGE_W), dtype=np.uint8)
for root, subdirs, files in os.walk(IMAGE_DIR):
	for file in files:
		path = root + "\\" + file
		if not (path.endswith('.jpg') or path.endswith('.png')):
			continue
		img = cv2.imread(path)
		
		if img is None:
			assert(False)
		if len(img.shape) != 3 or img.shape[2] != 3:
			assert(False)
		if img.shape[0] < IMAGE_H or img.shape[1] < IMAGE_W:
			assert(False)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = yb_resize(img)
		for i in range(SAMPLES_PER_IMG):
			y_data[ix] = np.transpose(img, (2, 0, 1))
			x_data[ix] = rand_dots(img, i)
			if ix < SAMPLES_PER_IMG*16:
				outimg = x_data[ix][0]
				cv2.imwrite(r'D:/Compressed/FaceEditor-master/imgs/cargb{}.png'.format(str(ix)), outimg)

			ix += 1
			y_data[ix] = np.flip(y_data[ix - 1], axis=2)
			x_data[ix] = np.flip(x_data[ix - 1], axis=2)
			ix += 1
		# 	if i==0:
		# 		y_data = np.transpose(img, (2, 0, 1))
		# 		y_data = np.expand_dims(y_data,axis=0)
		# 		x_data = rand_dots(img, i)
		# 		x_data = np.expand_dims(x_data,axis=0)
		# 	else:
		# 		y_d = np.transpose(img, (2, 0, 1))
		# 		y_d = np.expand_dims(y_d,axis=0)
		# 		y_data = np.append(y_data,y_d,axis=0)
		# 		x_d = rand_dots(img, i)
		# 		x_d = np.expand_dims(x_d,axis=0)
		# 		x_data = np.append(x_data,x_d,axis=0)

		# 	# if ix < SAMPLES_PER_IMG*16:
		# 	# 	outimg = x_data[ix]
		# 	# 	cv2.imwrite(r'D:/Compressed/FaceEditor-master/imgs/cargb{}.png'.format(str(ix)), outimg)
		# 	ix += 1
		# 	y_flip = np.flip(y_data[ix - 1], axis=2)
		# 	y_f = np.expand_dims(y_flip,axis=0)
		# 	y_data = np.append(y_data,y_f,axis=0)
		# 	x_flip = np.flip(x_data[ix - 1], axis=1)
		# 	x_f = np.expand_dims(x_flip,axis=0)
		# 	x_data = np.append(x_data,x_f,axis=0)	
		# 	ix += 1
			

		sys.stdout.write('\r')
		progress = ix * 100 / NUM_SAMPLES
		sys.stdout.write(str(progress) + "%\n")
		sys.stdout.flush()
		# assert(ix <= NUM_SAMPLES)


# print("Saving...")
# np.save('x_data.npy', x_data)
# np.save('y_data.npy', y_data)
print(x_data.shape)
print(y_data.shape)
print("hurray!!!")