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
 
#https://www.tensorflow.org/tutorials/generative/pix2pix
#https://github.com/robinvvinod/unet/blob/master/layers2D.py

#alpha = 0.1
#initializers ? = 'he_normal' 'glorot_uniform' tf.random_normal_initializer(0., 0.02)
 
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

initializer = 'he_normal'

def conv2d_block(input_tensor, filters, size, apply_batchnorm=True, 
			     strides=1, dilation_rate=1, recurrent=1):

	# A wrapper of the Keras Conv2D block to serve as a building block for downsampling layers
	# Includes options to use batch normalization, dilation and recurrence
	

	conv = Conv2D(filters=filters, kernel_size=size, strides=strides,
	              kernel_initializer=initializer, padding='same', dilation_rate=dilation_rate)(input_tensor)
	if apply_batchnorm :
		conv = BatchNormalization()(conv)
	output = LeakyReLU(alpha=0.1)(conv)

	for _ in range(recurrent - 1):
		conv = Conv2D(filters=filters, kernel_size=size, strides=1, 
	                  kernel_initializer=initializer, padding='same', dilation_rate=dilation_rate)(output)
		if apply_batchnorm :
			conv = BatchNormalization()(conv)
		res = LeakyReLU(alpha=0.1)(conv)
		output = Add()([output, res])
	
	return output


def residual_block(input_tensor, filters, size=3, strides=1, apply_batchnorm=True, 
                   recurrent=1, dilation_rate=1):

	# A residual block based on the ResNet architecture incorporating use of short-skip connections
	# Uses two successive convolution layers by default

	res = conv2d_block(input_tensor, filters=filters, size=size, strides=strides, 
	                   apply_batchnorm=apply_batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)
	res = conv2d_block(res, filters=filters, size=size, strides=1, 
	                   apply_batchnorm=apply_batchnorm, dilation_rate=dilation_rate, recurrent=recurrent)
	shortcut = conv2d_block(input_tensor, filters=filters, size=1, strides=strides, 
	                   apply_batchnorm=apply_batchnorm, dilation_rate=1, recurrent=recurrent)
	if apply_batchnorm:
		shortcut = BatchNormalization()(shortcut)
	output = Add()([shortcut, res])
	return output

def transpose_block(input_tensor, skip_tensor,
                    filters, size=3, strides=1,
                    apply_batchnorm=True, recurrent=1):

	# A wrapper of the Keras Conv2DTranspose block to serve as a building block for upsampling layers

	shape_x = K.int_shape(input_tensor)
	shape_xskip = K.int_shape(skip_tensor)

	conv = Conv2DTranspose(filters=filters,
                           kernel_size=size,
                           padding='same',
                           strides=(shape_xskip[1] // shape_x[1],
                                    shape_xskip[2] // shape_x[2]),
                           kernel_initializer="he_normal")(input_tensor)
	conv = LeakyReLU(alpha=0.1)(conv)

	act = conv2d_block(conv,
                       filters=filters,
                       size=size,
                       strides=1,
                       apply_batchnorm=apply_batchnorm,
                       dilation_rate=1,
                       recurrent=recurrent)
	output = Concatenate(axis=3)([act, skip_tensor])
	return output


def expend_as(tensor, rep):

	# Anonymous lambda function to expand the specified axis by a factor of argument, rep.
	# If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
	return my_repeat

def AttnGatingBlock(x, g, inter_shape):

	shape_x = K.int_shape(x)
	shape_g = K.int_shape(g)

	# Getting the gating signal to the same number of filters as the inter_shape
	phi_g = Conv2D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

	# Getting the x signal to the same shape as the gating signal
	theta_x = Conv2D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2]),
                     padding='same')(x)

	# Element-wise addition of the gating and x signals
	add_xg = add([phi_g, theta_x])
	add_xg = Activation('relu')(add_xg)

	# 1x1x1 convolution
	psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
	psi = Activation('sigmoid')(psi)
	shape_sigmoid = K.int_shape(psi)

	# Upsampling psi back to the original dimensions of x signal
	upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1],
                                             shape_x[2] //
                                             shape_sigmoid[2]))(psi)

	# Expanding the filter axis to the number of filters in the original x signal
	upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[3])

	# Element-wise multiplication of attention coefficients back onto original x signal
	attn_coefficients = multiply([upsample_sigmoid_xg, x])

	# Final 1x1x1 convolution to consolidate attention signal to original x dimensions
	output = Conv2D(filters=shape_x[3],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
	output = BatchNormalization()(output)
	return output


"""
def GatingSignal(input_tensor, batchnorm=True):

	# 1x1x1 convolution to consolidate gating signal into the required dimensions
	# Not required most of the time, unless another ReLU and batch_norm is required on gating signal

	shape = K.int_shape(input_tensor)
	conv = Conv2D(filters=shape[3],
                  kernel_size=1,
                  strides=1,
                  padding="same",
                  kernel_initializer="he_normal")(input_tensor)
	if batchnorm:
		conv = BatchNormalization()(conv)
	output = LeakyReLU()(conv)
	return output
"""


"""
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
	#last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer)

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
"""


def UNet(height, width, input_channels, output_channels):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d0 = conv2d_block(inputs, 16, size=3, strides=1) # 256*256
	d1 = conv2d_block(d0, 32, size=3, strides=2) # 128*128
	d2 = conv2d_block(d1, 64, size=3, strides=2) # 64*64
	d3 = conv2d_block(d2, 128, size=3, strides=2) # 32*32
	b0 = conv2d_block(d3, 256, size=3, strides=2) # 16*16

	u0 = transpose_block(b0, d3, 128) # 32*32
	u1 = transpose_block(u0, d2, 64) # 64*64
	u2 = transpose_block(u1, d1, 32) # 128*128
	u3 = transpose_block(u2, d0, 16) # 256*256

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(u3)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def R2UNet(height, width, input_channels, output_channels, recurrent=2):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d0 = residual_block(inputs, 16, size=3, strides=1, recurrent=recurrent) # 256*256
	d1 = residual_block(d0, 32, size=3, strides=2, recurrent=recurrent) # 128*128
	d2 = residual_block(d1, 64, size=3, strides=2, recurrent=recurrent) # 64*64
	d3 = residual_block(d2, 128, size=3, strides=2, recurrent=recurrent) # 32*32
	b0 = residual_block(d3, 256, size=3, strides=2, recurrent=recurrent) # 16*16

	u0 = transpose_block(b0, d3, 128, recurrent=recurrent) # 32*32
	u1 = transpose_block(u0, d2, 64, recurrent=recurrent) # 64*64
	u2 = transpose_block(u1, d1, 32, recurrent=recurrent) # 128*128
	u3 = transpose_block(u2, d0, 16, recurrent=recurrent) # 256*256

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(u3)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def AttUNet(height, width, input_channels, output_channels):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d0 = conv2d_block(inputs, 16, size=3, strides=1) # 256*256
	d1 = conv2d_block(d0, 32, size=3, strides=2) # 128*128
	d2 = conv2d_block(d1, 64, size=3, strides=2) # 64*64
	d3 = conv2d_block(d2, 128, size=3, strides=2) # 32*32
	b0 = conv2d_block(d3, 256, size=3, strides=2) # 16*16

	attn0 = AttnGatingBlock(d3, b0, 256)
	u0 = transpose_block(b0, attn0, 128) # 32*32
	attn1 = AttnGatingBlock(d2, u0, 128)
	u1 = transpose_block(u0, attn1, 64) # 64*64
	attn2 = AttnGatingBlock(d1, u1, 64)
	u2 = transpose_block(u1, attn2, 32) # 128*128
	u3 = transpose_block(u2, d0, 16) # 256*256

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(u3)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

	return tf.keras.Model(inputs=inputs, outputs=x)
