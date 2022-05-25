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
 
# https://www.tensorflow.org/tutorials/images/segmentation
# https://www.tensorflow.org/tutorials/generative/pix2pix
# https://github.com/robinvvinod/unet
# https://github.com/ykamikawa/tf-keras-SegNet
# https://www.kaggle.com/code/meaninglesslives/pretrained-resnet34-in-keras/notebook

#alpha = 0.1
#initializers ? = 'he_normal' 'glorot_uniform' tf.random_normal_initializer(0., 0.02)
 
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

initializer = 'glorot_uniform'


#############################################################################################
#							        BRICKS                                                  #
#############################################################################################




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
	shape_xskip = K.int_shape(skip_tensor[0])

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
	output = Concatenate(axis=3)([act] + skip_tensor)
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


def inception_block(input_tensor, filters, size=3, strides=1, apply_batchnorm=True, recurrent=1, layers=[]):

	# Inception-style convolutional block similar to InceptionNet
	# The first convolution follows the function arguments, while subsequent inception convolutions follow the parameters in
	# argument, layers

	# layers is a nested list containing the different secondary inceptions in the format of (kernel_size, dil_rate)

	# E.g => layers=[ [(3,1),(3,1)], [(5,1)], [(3,1),(3,2)] ]
	# This will implement 3 sets of secondary convolutions
	# Set 1 => 3x3 dil = 1 followed by another 3x3 dil = 1
	# Set 2 => 5x5 dil = 1
	# Set 3 => 3x3 dil = 1 followed by 3x3 dil = 2

	res = conv2d_block(input_tensor, filters, size, strides, apply_batchnorm, 1, recurrent)

	temp = []
	for layer in layers:
		local_res = res
		for conv in layer:
			incep_kernel_size = conv[0]
			incep_dilation_rate = conv[1]
			local_res = conv2d_block(local_res, filters, incep_kernel_size, 1, apply_batchnorm, incep_dilation_rate, recurrent)
		temp.append(local_res)
	
	conc = Concatenate()(temp)
	tf.print(conc.shape)
	res = conv2d_block(conc, filters, size, 1, apply_batchnorm, 1)

	shortcut = conv2d_block(input_tensor, filters, size, strides, apply_batchnorm, 1)
	if apply_batchnorm :
		shortcut = BatchNormalization()(shortcut)

	output = Add()([shortcut, res])
	return output

def MaxPoolingWithArgmax2D(input_tensor, pool=(2,2), strides=(2,2), padding="SAME"):
	ksize = [1, pool[0], pool[1], 1]
	strides = [1, strides[0], strides[1], 1]
	output, argmax = tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding=padding)

	return [output, argmax]

def MaxUnpooling2D(inputs, output_shape=None, size=(2,2)):
	updates, mask = inputs[0], inputs[1]
	mask = K.cast(mask, "int32")
	input_shape = tf.shape(updates, out_type="int32")
	#print(input_shape)
	if output_shape is None:
		output_shape = (input_shape[0], input_shape[1]*size[0], input_shape[2]*size[1], input_shape[3])
	
	# calculation indices for batch, height, width and feature maps
	one_like_mask = K.ones_like(mask, dtype="int32")
	batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
	batch_range = K.reshape(tf.range(output_shape[0], dtype="int32"), shape=batch_shape)
	b = one_like_mask * batch_range
	#print(output_shape)
	y = mask // (output_shape[2] * output_shape[3])
	x = (mask // output_shape[3]) % output_shape[2]
	feature_range = tf.range(output_shape[3], dtype="int32")
	f = one_like_mask * feature_range

	# transpose indices & reshape update values to one dimension
	updates_size = tf.size(updates)
	indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
	values = K.reshape(updates, [updates_size])
	ret = tf.scatter_nd(indices, values, output_shape)
	print(ret)
	return ret


#############################################################################################
#							        MODELS                                                  #
#############################################################################################

# paper version
def UNet(height, width, input_channels, output_channels):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d1 = conv2d_block(inputs, 64, size=4, strides=2) # 128*128
	d2 = conv2d_block(d1, 128, size=4, strides=2) # 64*64
	d3 = conv2d_block(d2, 256, size=4, strides=2) # 32*32
	d4 = conv2d_block(d2, 512, size=4, strides=2) # 16*16
	d5 = conv2d_block(d2, 512, size=4, strides=2) # 8*8
	b0 = conv2d_block(d3, 512, size=4, strides=2) # 4*4

	u0 = transpose_block(b0, [d5], 512, size=4) # 8*8
	u1 = transpose_block(u0, [d4], 512, size=4) # 16*16
	u2 = transpose_block(u1, [d3], 256, size=4) # 32*32
	u3 = transpose_block(u2, [d2], 128, size=4) # 64*64
	u4 = transpose_block(u3, [d1], 64, size=4) # 128*128

	outputs = Conv2DTranspose(filters=output_channels, kernel_size=4, strides=2, padding='same')(u4)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def R2UNet(height, width, input_channels, output_channels, recurrent=2):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d0 = residual_block(inputs, 16, size=3, strides=1, recurrent=recurrent) # 256*256
	d1 = residual_block(d0, 32, size=3, strides=2, recurrent=recurrent) # 128*128
	d2 = residual_block(d1, 64, size=3, strides=2, recurrent=recurrent) # 64*64
	d3 = residual_block(d2, 128, size=3, strides=2, recurrent=recurrent) # 32*32
	b0 = residual_block(d3, 256, size=3, strides=2, recurrent=recurrent) # 16*16

	u0 = transpose_block(b0, [d3], 128, recurrent=recurrent) # 32*32
	u1 = transpose_block(u0, [d2], 64, recurrent=recurrent) # 64*64
	u2 = transpose_block(u1, [d1], 32, recurrent=recurrent) # 128*128
	u3 = transpose_block(u2, [d0], 16, recurrent=recurrent) # 256*256

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
	u0 = transpose_block(b0, [attn0], 128) # 32*32
	attn1 = AttnGatingBlock(d2, u0, 128)
	u1 = transpose_block(u0, [attn1], 64) # 64*64
	attn2 = AttnGatingBlock(d1, u1, 64)
	u2 = transpose_block(u1, [attn2], 32) # 128*128
	u3 = transpose_block(u2, [d0], 16) # 256*256

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(u3)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def InceptionUNet(height, width, input_channels, output_channels):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	d0 = inception_block(inputs, 16, size=3, strides=1, layers=[[(3, 1)]]) # 256*256 # [(3, 1), (3, 1)], [(3, 2)] => not enough memory
	d1 = inception_block(d0, 32, size=3, strides=2, layers=[[(3, 1)]]) # 128*128
	d2 = inception_block(d1, 64, size=3, strides=2, layers=[[(3, 1)]]) # 64*64
	d3 = inception_block(d2, 128, size=3, strides=2, layers=[[(3, 1)]]) # 32*32
	b0 = inception_block(d3, 256, size=3, strides=2, layers=[[(3, 1)]]) # 16*16

	u0 = transpose_block(b0, [d3], 128) # 32*32
	u1 = transpose_block(u0, [d2], 64) # 64*64
	u2 = transpose_block(u1, [d1], 32) # 128*128
	u3 = transpose_block(u2, [d0], 16) # 256*256

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(u3)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def UnetPlusPlus(height, width, input_channels, output_channels):

	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	x00 = conv2d_block(inputs, 16, size=3, strides=1) # 256*256
	x10 = conv2d_block(x00, 32, size=3, strides=2) # 128*128
	x20 = conv2d_block(x10, 64, size=3, strides=2) # 64*64
	x30 = conv2d_block(x20, 128, size=3, strides=2) # 32*32
	x40 = conv2d_block(x30, 256, size=3, strides=2) # 16*16

	x01 = transpose_block(x10, [x00], 16) # 256*256
	x11 = transpose_block(x20, [x10], 32) # 128*128
	x21 = transpose_block(x30, [x20], 64) # 64*64
	x31 = transpose_block(x40, [x30], 128) # 32*32

	x02 = transpose_block(x11, [x00, x01], 16) # 256*256 
	x12 = transpose_block(x21, [x10, x11], 32) # 128*128
	x22 = transpose_block(x31, [x20, x21], 32) # 64*64

	x03 = transpose_block(x12, [x00, x01, x02], 16) # 256*256
	x13 = transpose_block(x22, [x10, x11, x12], 32) # 128*128

	x04 = transpose_block(x13, [x00, x01, x02, x03], 16) # 256*256 

	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(x04)

	return tf.keras.Model(inputs=inputs, outputs=outputs)


# basic version with upsampling layers. need a way to make MaxUnpooling2D work properly
def SegNet(height, width, input_channels, output_channels):

    # encoder
	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	conv_1 = conv2d_block(inputs, 64, 3)
	conv_2 = conv2d_block(conv_1, 64, 3)

	#pool_1, mask_1 = MaxPoolingWithArgmax2D(conv_2, (2,2))# 128*128
	pool_1 = MaxPool2D(2)(conv_2) # 128*128

	conv_3 = conv2d_block(pool_1, 128, 3)
	conv_4 = conv2d_block(conv_3, 128, 3)

	#pool_2, mask_2 = MaxPoolingWithArgmax2D(conv_4, (2,2)) # 64*64
	pool_2 = MaxPool2D(2)(conv_4) # # 64*64

	conv_5 = conv2d_block(pool_2, 256, 3)
	conv_6 = conv2d_block(conv_5, 256, 3)
	conv_7 = conv2d_block(conv_6, 256, 3)

	#pool_3, mask_3 = MaxPoolingWithArgmax2D(conv_7, (2,2)) # 32*32
	pool_3 = MaxPool2D(2)(conv_7) # 32*32

	conv_8 = conv2d_block(pool_3, 512, 3) 
	conv_9 = conv2d_block(conv_8, 512, 3)
	conv_10 = conv2d_block(conv_9, 512, 3)

	#pool_4, mask_4 = MaxPoolingWithArgmax2D(conv_10, (2,2)) # 16*16
	pool_4 = MaxPool2D(2)(conv_10) # 16*16

	conv_11 = conv2d_block(pool_4, 512, 3)
	conv_12 = conv2d_block(conv_11, 512, 3)
	conv_13 = conv2d_block(conv_12, 512, 3)

	#pool_5, mask_5 = MaxPoolingWithArgmax2D(conv_13, (2,2)) # 8*8
	pool_5 = MaxPool2D(2)(conv_13) # 8*8

	# decoder

	#unpool_1 = MaxUnpooling2D([pool_5, mask_5], size=(2,2)) # 16*16
	unpool_1 = UpSampling2D(2)(pool_5) # 16*16

	conv_14 = conv2d_block(unpool_1, 512, 3)
	conv_15 = conv2d_block(conv_14, 512, 3)
	conv_16 = conv2d_block(conv_15, 512, 3)

	#unpool_2 = MaxUnpooling2D([conv_16, mask_4], size=(2,2)) # 32*32
	unpool_2 = UpSampling2D(2)(conv_16) # 32*32

	conv_17 = conv2d_block(unpool_2, 512, 3)
	conv_18 = conv2d_block(conv_17, 512, 3)
	conv_19 = conv2d_block(conv_18, 512, 3)

	#unpool_3 = MaxUnpooling2D([conv_19, mask_3], size=(2,2)) # 64*64
	unpool_3 = UpSampling2D(2)(conv_19) # 64*64

	conv_20 = conv2d_block(unpool_3, 128, 3)
	conv_21 = conv2d_block(conv_20, 128, 3)
	conv_22 = conv2d_block(conv_21, 128, 3)

	#unpool_4 = MaxUnpooling2D([conv_22, mask_2], size=(2,2)) # 128*128
	unpool_4 = UpSampling2D(2)(conv_22) # 128*128

	conv_23 = conv2d_block(unpool_4, 64, 3)
	conv_24 = conv2d_block(conv_23, 64, 3)

	#unpool_5 = MaxUnpooling2D([conv_24, mask_1], size=(2,2)) #256*256
	unpool_5 = UpSampling2D(2)(conv_24) # 256*256

	conv_25 = conv2d_block(unpool_5, 64, 3)
	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(conv_25)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SegNet")

	return model

# basic version with upsampling layers. need a way to make MaxUnpooling2D work properly
def SegNetPreTrained(height, width, input_channels, output_channels):

	base_model = tf.keras.applications.vgg16.VGG16(input_shape=[256, 256, 3], include_top=False, pooling=max)
	layers = ['block1_pool',
	          'block2_pool',
			  'block3_pool',
			  'block4_pool',
			  'block5_pool'
	]
	base_model_outputs = [base_model.get_layer(name).output for name in layers]

	# Create the feature extraction model
	down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
	down_stack.trainable = False
	
	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	# Downsampling through the model
	skips = down_stack(inputs)
	x = skips[-1]

	# decoder

	#unpool_1 = MaxUnpooling2D([pool_5, mask_5], size=(2,2)) # 16*16
	unpool_1 = UpSampling2D(2)(x) # 16*16

	conv_14 = conv2d_block(unpool_1, 512, 3)
	conv_15 = conv2d_block(conv_14, 512, 3)
	conv_16 = conv2d_block(conv_15, 512, 3)

	#unpool_2 = MaxUnpooling2D([conv_16, mask_4], size=(2,2)) # 32*32
	unpool_2 = UpSampling2D(2)(conv_16) # 32*32

	conv_17 = conv2d_block(unpool_2, 512, 3)
	conv_18 = conv2d_block(conv_17, 512, 3)
	conv_19 = conv2d_block(conv_18, 512, 3)

	#unpool_3 = MaxUnpooling2D([conv_19, mask_3], size=(2,2)) # 64*64
	unpool_3 = UpSampling2D(2)(conv_19) # 64*64

	conv_20 = conv2d_block(unpool_3, 128, 3)
	conv_21 = conv2d_block(conv_20, 128, 3)
	conv_22 = conv2d_block(conv_21, 128, 3)

	#unpool_4 = MaxUnpooling2D([conv_22, mask_2], size=(2,2)) # 128*128
	unpool_4 = UpSampling2D(2)(conv_22) # 128*128

	conv_23 = conv2d_block(unpool_4, 64, 3)
	conv_24 = conv2d_block(conv_23, 64, 3)

	#unpool_5 = MaxUnpooling2D([conv_24, mask_1], size=(2,2)) #256*256
	unpool_5 = UpSampling2D(2)(conv_24) # 256*256

	conv_25 = conv2d_block(unpool_5, 64, 3)
	outputs = Conv2D(filters=output_channels, kernel_size=1, strides=1)(conv_25)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SegNet")

	return model

def UResNet50PreTrained(height, width, input_channels, output_channels):

	base_model = tf.keras.applications.ResNet50(input_shape=[256, 256, 3], include_top=False, pooling=max)

	layers = ['conv1_relu',
	          'conv2_block3_out',
			  'conv3_block4_out',
			  'conv4_block6_out',
			  'conv5_block3_out'
	]
	base_model_outputs = [base_model.get_layer(name).output for name in layers]

	#for layer in base_model.layers :
	#	print(layer.name)

	# Create the feature extraction model
	down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
	down_stack.trainable = False
	
	inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

	# Downsampling through the model
	skips = down_stack(inputs) 

	# decoder

	u0 = transpose_block(skips[4], [skips[3]], 512, size=4) # 16*16
	u1 = transpose_block(u0, [skips[2]], 512, size=4) # 32*32
	u2 = transpose_block(u1, [skips[1]], 256, size=4) # 64*64
	u3 = transpose_block(u2, [skips[0]], 128, size=4) # 128*128

	outputs = Conv2DTranspose(filters=output_channels, kernel_size=4, strides=2, padding='same')(u3)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="UResNet50")

	return model
