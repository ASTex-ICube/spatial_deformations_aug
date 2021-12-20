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

def downsample(filters, size, apply_batchnorm=True):
	initializer = 'glorot_uniform' #tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001, scale=True))

	result.add(tf.keras.layers.LeakyReLU())

	return result

def upsample(filters, size, apply_dropout=False):
	initializer = 'glorot_uniform' #tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

	result.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.00001, scale=True))

	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

def UNet(height, width, input_channels, output_channels):
	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64) apply_batchnorm=False
		downsample(128, 4),                        # (batch_size, 64, 64, 128)
		downsample(256, 4),                        # (batch_size, 32, 32, 256)
		downsample(512, 4),                        # (batch_size, 16, 16, 512)
		downsample(512, 4),                        # (batch_size, 8, 8, 512)
		downsample(512, 4),                        # (batch_size, 4, 4, 512)
		#downsample(512, 4),                        # (batch_size, 2, 2, 512)
		#downsample(512, 4),                        # (batch_size, 1, 1, 512)
		]
		  
	up_stack = [
		#upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
		#upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
		upsample(512, 4),                      # (batch_size, 16, 16, 1024)
		upsample(256, 4),                      # (batch_size, 32, 32, 512)
		upsample(128, 4),                      # (batch_size, 64, 64, 256)
		upsample(64, 4),                       # (batch_size, 128, 128, 128)
		]

	initializer = 'glorot_uniform' #tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer)

	x = inputs

	# Downsampling through the model
	skips = []
	for down in down_stack :
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	# Up sampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)