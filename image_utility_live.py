'''
 * Data augmentation based on random spatial deformations
 * Authors: F. Allender, R. Allègre, C. Wemmert, J.-M. Dischler
 *
 * Code author: Florian Allender
 *
 * anonymous
 * anonymous

 * @version 1.0
'''

import tensorflow as tf
import tensorflow_addons as tfa

def normalize(image):

	image = image / (tf.math.reduce_max(image)+tf.keras.backend.epsilon())
	#tf.print(tf.math.reduce_min(image), tf.math.reduce_max(image))

	return image

def load(image_file, channels):

	# Read and decode an image file to a uint8 tensor
	image = tf.io.read_file(image_file)
	image = tf.image.decode_png(image, channels=channels)

	# Convert image to float32 tensor
	image = tf.cast(image, tf.float32)

	# Normalizing the image to [0, 1]
	image = normalize(image)

	return image


def load_images(image_file, gt_file, input_channels, output_channels, augment=False, n=3, sigma=20):

	image_batch = []
	gt_batch = []

	for i in range(len(image_file)):

		image = load(image_file[i], input_channels)
		gt = load(gt_file[i], output_channels)

		if augment :
			image, gt = augment_pair(image, gt, n, sigma)

		image_batch.append(image)
		gt_batch.append(gt)

	return tf.stack(image_batch, axis=0), tf.stack(gt_batch, axis=0)

def expand(x,y):

	return (tf.expand_dims(x, axis=0), tf.expand_dims(y, axis=0))

def create_dataset(dataset_name, subset):

	dataset_img = tf.data.Dataset.list_files(dataset_name + '/' + subset + '/images/*.png', shuffle = False)
	dataset_gt = tf.data.Dataset.list_files(dataset_name + '/' + subset + '/gts/*.png', shuffle = False)
	dataset = tf.data.Dataset.zip((dataset_img, dataset_gt))

	return dataset


def augment_pair(image, gt, n, sigma):

	if tf.random.uniform([], minval=0, maxval=1) > 0.5:
		image = tf.image.rot90(image)
		gt    = tf.image.rot90(gt)
	if tf.random.uniform([], minval=0, maxval=1) > 0.5:
		image = tf.image.flip_up_down(image)
		gt    = tf.image.flip_up_down(gt)
	if tf.random.uniform([], minval=0, maxval=1) > 0.5:
		image = tf.image.flip_left_right(image)
		gt    = tf.image.flip_left_right(gt)
	if tf.random.uniform([], minval=0, maxval=1) > 0.25:
		image, gt = elt(image, gt, n, sigma)
	#print(image.shape[0])

	return image, gt

def elt(image, gt, n=3, sigma=20):

	width = image.shape[0]
	height = image.shape[1]

	pad_size = 128

	displacement = tf.random.normal([n*n, 2], 0.0, sigma, dtype=tf.float32)
	
	x = tf.linspace(0.0 + pad_size, height * 1.0 + pad_size, n)
	y = tf.linspace(0.0 + pad_size, width * 1.0 + pad_size, n)
	X, Y = tf.meshgrid(x, y)

	source = tf.stack([Y, X], axis=2)
	source = tf.reshape(source, [n*n, 2])
	dest = tf.add(source, displacement)
				
	source = tf.expand_dims(source, axis=0)
	dest = tf.expand_dims(dest, axis=0)
	image = tf.expand_dims(image, axis=0)
	gt = tf.expand_dims(gt, axis=0)

	# Add padding
	paddings = tf.constant([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])
	image = tf.pad(image[0,:,:,:], paddings, "REFLECT")
	gt = tf.pad(gt[0,:,:,:], paddings, "REFLECT")
	
	# Warp image
	# https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp
	dense_img_warp, flow_field = tfa.image.sparse_image_warp(image, source, dest,
	                                                      	num_boundary_points=0,
														 	interpolation_order=3)
	dense_gt_warp, flow_field = tfa.image.sparse_image_warp(gt, source, dest,
	                                                     	num_boundary_points=0,
														 	interpolation_order=3)

	# Remove padding
	dense_img_warp = tf.image.crop_to_bounding_box(dense_img_warp, pad_size, pad_size, height, width)
	dense_gt_warp = tf.image.crop_to_bounding_box(dense_gt_warp, pad_size, pad_size, height, width)

	# Remove batch dimension
	#image = tf.squeeze(dense_img_warp, 0)
	#gt = tf.squeeze(dense_gt_warp, 0)

	return dense_img_warp, dense_gt_warp
