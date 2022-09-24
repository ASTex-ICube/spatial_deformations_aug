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

import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def callback(results, dataset_name, file_name, method):
	file = open(file_name, "a")
	file.write("-----------------------------------------------------\n")
	file.write("\n")
	file.write(dataset_name + " " + method + "\n")
	string = "[Test Dice score = %f]\n" % results[0]
	string = string + "[Test Prec = %f] [Test Recall = %f] [Test Spec = %f]\n" % (results[1], results[3], results[4])
	file.write(string)
	file.close()

def training_curves(train_points, validation_points, dataset_name):

	plt.figure()

	plt.plot(train_points, "r--", label = 'train loss')
	plt.plot(validation_points, "b--", label = 'validation loss')

	plt.xlabel("Epochs")
	plt.ylabel("Losses")
	#plt.title("Train and Validation Losses Over Epochs", fontsize=14)
	plt.legend()

	if len(dataset_name.split('/')) == 1:
		save_name = dataset_name.split('/')[0]
	else:
		save_name = dataset_name.split('/')[1]
	plt.savefig('plots/'+save_name+'.png')
	plt.close()

def display(display_list, n=0, j=0):
	plt.figure()

	title = ['input', 'true mask', 'predicted']

	for i in range(len(display_list)):
		#print(display_list[i])
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.savefig('plots/'+'callback_'+str(n)+'_'+str(j)+'.png')
	plt.close()

def create_mask(pred_mask):

	probas = tf.math.sigmoid(pred_mask[0,:,:])
	bw = np.zeros_like(probas)
	bw[probas > 0.5] = 1.0
	
	return bw
