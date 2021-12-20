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

def normalize(image):

	image = image / (tf.math.reduce_max(image)+tf.keras.backend.epsilon())
	#tf.print(tf.math.reduce_min(image), tf.math.reduce_max(image))

	return image

def load(image_file, height, width, channels):

	# Read and decode an image file to a uint8 tensor
	image = tf.io.read_file(image_file)
	image = tf.image.decode_png(image, channels=channels)

	# Convert image to float32 tensor
	image = tf.cast(image, tf.float32)

	# Normalizing the image to [0, 1]
	image = normalize(image)

	return image


def load_images(image_file, gt_file, height, width, input_channels, output_channels):

	image = load(image_file, height, width, input_channels)
	gt = load(gt_file, height, width, output_channels)

	return (image, gt)

def expand(x,y):

	return (tf.expand_dims(x, axis=0), tf.expand_dims(y, axis=0))

def create_dataset(dataset_name, subset, height, width, input_channels, output_channels, augment=False):

	dataset_img = tf.data.Dataset.list_files(dataset_name + '/' + subset + '/images/*.png', shuffle = False)
	dataset_gt = tf.data.Dataset.list_files(dataset_name + '/' + subset + '/gts/*.png', shuffle = False)
	dataset = tf.data.Dataset.zip((dataset_img, dataset_gt))

	dataset = dataset.map(lambda x,y : load_images(x, y, height, width, input_channels, output_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)

	if augment:
		dataset = augment_dataset(dataset)
		dataset = dataset.unbatch()
		
	#dataset = dataset.map(expand, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	return dataset

def augment_image(image, gt):

	augmented_image = [image]
	augmented_gt = [gt]

	image_rot = tf.image.rot90(image)
	augmented_image.append(image_rot)
	gt_rot = tf.image.rot90(gt)
	augmented_gt.append(gt_rot)

	for j in range(2):
		image_rot = tf.image.rot90(image_rot)
		augmented_image.append(image_rot)
		gt_rot = tf.image.rot90(gt_rot)
		augmented_gt.append(gt_rot)

	augmented_image.append(tf.image.flip_up_down(augmented_image[0]))
	augmented_image.append(tf.image.flip_up_down(augmented_image[1]))
	augmented_image.append(tf.image.flip_left_right(augmented_image[0]))
	augmented_image.append(tf.image.flip_left_right(augmented_image[1]))

	augmented_gt.append(tf.image.flip_up_down(augmented_gt[0]))
	augmented_gt.append(tf.image.flip_up_down(augmented_gt[1]))
	augmented_gt.append(tf.image.flip_left_right(augmented_gt[0]))
	augmented_gt.append(tf.image.flip_left_right(augmented_gt[1]))

	return (augmented_image, augmented_gt)

def augment_dataset(dataset):

	dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	return dataset
