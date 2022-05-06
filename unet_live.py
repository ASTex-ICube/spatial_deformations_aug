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
import numpy as np

import os
import datetime

import parser
import image_utility_live as iu
import build_network as bn
import helper_live as h

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def crossentropy_loss(gt_image, seg_output, loss_object):

	loss = loss_object(gt_image, seg_output)

	return loss

#@tf.function
def train_step(input_image, gt_image, epoch, unet, loss_object, optimizer):

	with tf.GradientTape() as unet_tape:
		seg_output = unet(input_image, training=True)
		loss = crossentropy_loss(gt_image, seg_output, loss_object)

		#display_list = [input_image[0], gt_image[0], h.create_mask(seg_output[0])]
		#h.display(display_list, epoch)

	unet_gradients = unet_tape.gradient(loss, unet.trainable_variables)

	optimizer.apply_gradients(zip(unet_gradients, unet.trainable_variables))

	return loss

def validation_step(validation_dataset, epoch, epochs, unet, loss_object, verbose):

	validation_loss = []
	for n, (image_files, gt_files) in validation_dataset.enumerate():

		input_image, gt_image = iu.load_images(image_files, gt_files, 3, 1, augment=False)
		seg_output = unet(input_image, training=False)
		loss = crossentropy_loss(gt_image, seg_output, loss_object)
		if verbose:
			print("Validation [Epoch %d/%d] [Batch %d]" % (epoch+1, epochs, n+1))
			print("[Loss = %f]" % (loss.numpy()))
		validation_loss.append(loss)

	return np.mean(validation_loss)

def test(test_dataset, unet, batch_size, verbose, threshold=0.5):

	dice_n   = 0
	prec_n   = 0
	iou_n    = 0
	recall_n = 0
	spec_n   = 0
	dice_d   = 0
	prec_d   = 0
	iou_d    = 0
	recall_d = 0
	spec_d   = 0

	for n, (image_files, gt_files) in test_dataset.enumerate():

		input_image, gt_image = iu.load_images(image_files, gt_files, 3, 1, augment=False)

		seg_output = unet(input_image, training=False)

		for i in range(batch_size):
			probas = tf.math.sigmoid(seg_output[i,:,:,0])
			bw = np.zeros_like(probas)
			bw[probas > threshold] = 1.0

			A = bw
			B = gt_image[i,:,:,0]

			TP = np.sum(A*B)                # true positives
			U  = np.sum(A) + np.sum(B)      # union
			P  = np.sum(B)                  # gt positives
			FP = np.sum(A) - TP             # false positives
			FN = np.sum(B) - TP             # false negatives
			TN = np.sum(A==B) - np.sum(A*B) # true negatives

			dice_n   = dice_n + 2*TP
			dice_d   = dice_d + U
			prec_n   = prec_n + TP
			prec_d   = prec_d + TP + FP  
			iou_n    = iou_n + TP
			iou_d    = iou_d + TP + FP + FN
			recall_n = recall_n + TP
			recall_d = recall_d + TP + FN
			spec_n   = spec_n + TN
			spec_d   = spec_d + TN + FP

			"""
			save_images = "images"
			mpimg.imsave("%s/seg_%d.png" % (save_images, (n+1)*i), np.asarray(bw))
			mpimg.imsave("%s/gt_%d.png" % (save_images, (n+1)*i), np.asarray(gt_image[i,:,:,0]))
			mpimg.imsave("%s/i_%d.png" % (save_images, (n+1)*i), np.asarray(input_image[i]))
			"""

		if verbose:				
			print("Testing [Batch %d / %d], threshold = %f" % (n+1, test_dataset.cardinality().numpy(), threshold))

	Dice_m   = dice_n / dice_d
	Prec_m   = prec_n / prec_d
	IOU_m    = iou_n / iou_d
	Recall_m = recall_n / recall_d
	Spec_m   = spec_n / spec_d
							
	print("[Test Dice score = %f]" % (Dice_m))
	print("[Test Prec = %f] [Test Recall = %f] [Test Spec = %f]" % (Prec_m, Recall_m, Spec_m))

	return [Dice_m, Prec_m, IOU_m, Recall_m, Spec_m]


def fit(train_dataset, validation_dataset, batch_size, epochs, verbose, n, sigma, unet, loss_object, optimizer, checkpoint, manager, dataset_name):

	start_time = datetime.datetime.now()
	start = int(checkpoint.step)//train_dataset.cardinality().numpy()

	min_loss = 100

	loss_tr_callback = []
	loss_val = []

	train_dataset = train_dataset.shuffle(train_dataset.cardinality(), reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

	for epoch in range(start, epochs):

		loss_tr = []

		# Training step
		for i, (image_files, gt_files) in train_dataset.enumerate():
			
			checkpoint.step.assign_add(1)

			input_image, gt_image = iu.load_images(image_files, gt_files, 3, 1, augment=True, n=n, sigma=sigma)			

			loss = train_step(input_image, gt_image, epoch, unet, loss_object, optimizer)

			if verbose:
				print("Training [Epoch %d/%d] [Batch %d] time : %s" % (epoch+1, epochs, i+1, datetime.datetime.now()-start_time))
				print("[Loss = %f]" % (loss.numpy()))

			loss_tr.append(loss)

		validation_loss = validation_step(validation_dataset, epoch, epochs, unet, loss_object, verbose)
		if verbose:
			print("[Validation loss = %f]" % (validation_loss))
		if validation_loss < min_loss :
			min_loss = validation_loss
			save_path = manager.save()					
			print("Saved checkpoint for step {}: {}, time: {}".format(int(checkpoint.step), save_path, datetime.datetime.now()-start_time))
					
		loss_tr_callback.append(np.mean(loss_tr))
		loss_val.append(validation_loss)
		h.training_curves(loss_tr_callback, loss_val, dataset_name)


def main(args):
	
	if not os.path.exists('./plots'):
		os.makedirs('./plots')

	#------------------ Hyperparameters --------------------#

	# Input / output shape

	height   = args.img_height
	width    = args.img_width
	input_channels  = 3
	output_channels = 1
	
	# Training hyperparameters
	
	learning_rate = args.learning_rate
	beta          = args.beta

	batch_size    = args.batch_size
	epochs        = args.epochs
	verbose       = args.verbose

	n = args.n_grid
	sigma = args.sigma

	# Create datasets

	dataset_path = '../Data/' + args.dataset_name
	train_dataset      = iu.create_dataset(dataset_path, 'train')

	dataset_path = '../Data/' + args.dataset_name.split('/')[0]
	validation_dataset = iu.create_dataset(dataset_path, 'validation')
	test_dataset       = iu.create_dataset(dataset_path, 'test')

	validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)
	test_dataset       = test_dataset.batch(batch_size, drop_remainder=True)

	# Define the model
	
	unet_type = args.unet_type

	if unet_type == 'unet':
		unet = bn.UNet(height, width, input_channels, output_channels)
	elif unet_type == 'R2unet':
		unet = bn.R2UNet(height, width, input_channels, output_channels)
	elif unet_type == 'attunet':
		unet = bn.AttUNet(height, width, input_channels, output_channels)
	unet.summary()

	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta)

	checkpoint_directory = "./training_checkpoints/" + args.dataset_name
	checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

	checkpoint = tf.train.Checkpoint(step=tf.Variable(0), n=tf.Variable(0), optimizer=optimizer, model=unet)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=1)

	if args.resume_training == True or args.test == True :
		checkpoint.restore(manager.latest_checkpoint)
		print("Restored from {}".format(manager.latest_checkpoint))

	# Train or test the model
	if args.test == True:
		results = test(test_dataset, unet, batch_size, verbose)
		h.callback(results, args.dataset_name, args.unet_type)
	else:
		fit(train_dataset, validation_dataset, batch_size, epochs, verbose, n, sigma, unet, loss_object, optimizer, checkpoint, manager, args.dataset_name)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
