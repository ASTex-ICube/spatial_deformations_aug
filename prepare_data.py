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

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob
import time
import random

import PIL
from PIL import Image
from skimage import img_as_ubyte

import tensorflow as tf
import tensorflow_addons as tfa
tf.executing_eagerly()

from skimage import io, img_as_float, img_as_ubyte


def fetch_file_names(file_dir):
	file_names = glob('%s/*.png' % file_dir)
	return np.array(file_names)
		  
# Function to produce height*width patches from large images
# (may be skipped, see below for usage)
def prepare_patches(N, nb_crops, subsets, orig_dir, height, width):

	for i in range(len(subsets)):

		subset = subsets[i]
		n = N[i]

		save_dir = '../Data/' + orig_dir + '_' + str(N[0]) + '/' + subset       

		if not os.path.exists('%s/images/' % save_dir):
			os.makedirs('%s/images/' % save_dir)
		if not os.path.exists('%s/gts/' % save_dir):
			os.makedirs('%s/gts/' % save_dir)

		images_glomeruli = fetch_file_names('../Data/' + orig_dir + '/%s/images/glomeruli' % subset)
		images_negative  = fetch_file_names('../Data/' + orig_dir + '/%s/images/negative' % subset)
		gts_glomeruli    = fetch_file_names('../Data/' + orig_dir + '/%s/gts/glomeruli' % subset)
		gts_negative     = fetch_file_names('../Data/' + orig_dir + '/%s/gts/negative' % subset)

		for i in range(n):

			print('Cropping image %d/%d' % (i+1, n))
			
			file_image_glomeruli = images_glomeruli[i]
			file_gt_glomeruli = gts_glomeruli[i]

			image_glomeruli = PIL.Image.open(file_image_glomeruli).copy()
			gt_glomeruli = PIL.Image.open(file_gt_glomeruli).copy()
			
			file_image_negative = images_negative[i]
			file_gt_negative = gts_negative[i]

			image_negative = PIL.Image.open(file_image_negative)
			gt_negative = PIL.Image.open(file_gt_negative)

			file_image_glomeruli_noext = os.path.splitext(file_image_glomeruli)[0].split('/')[-1]
			file_gt_glomeruli_noext    = os.path.splitext(file_gt_glomeruli)[0].split('/')[-1]
			file_image_negative_noext  = os.path.splitext(file_image_negative)[0].split('/')[-1]
			file_gt_negative_noext     = os.path.splitext(file_gt_negative)[0].split('/')[-1]

			for j in range(nb_crops):
				x = np.random.randint(0, height)
				y = np.random.randint(0, width)
				
				region = (x,y, x+height, y+width)

				image_glomeruli_cropped = image_glomeruli.crop(region)
				
				gt_glomeruli_cropped = gt_glomeruli.crop(region)
				gt_glomeruli_cropped_clipped = np.zeros_like(gt_glomeruli_cropped)
				gt_glomeruli_cropped_clipped[np.asarray(gt_glomeruli_cropped) > 0.0] = 1.0
				gt_glomeruli_cropped_clipped = img_as_ubyte(gt_glomeruli_cropped)

				image_negative_cropped = image_negative.crop(region)
				
				gt_negative_cropped = gt_negative.crop(region)

				io.imsave(save_dir + '/images/'
						  + file_image_glomeruli_noext + '_' + str(j)+'.png', np.asarray(image_glomeruli_cropped))
				io.imsave(save_dir + '/gts/'
						  + file_gt_glomeruli_noext + '_' + str(j)+'.png', np.asarray(gt_glomeruli_cropped_clipped), check_contrast=False)
				io.imsave(save_dir + '/images/'
						  + file_image_negative_noext + '_' + str(j)+'.png', np.asarray(image_negative_cropped))
				io.imsave(save_dir + '/gts/'
						  + file_gt_negative_noext + '_' + str(j)+'.png', np.asarray(gt_negative_cropped), check_contrast=False)

	return  orig_dir + '_' + str(N[0])

# Function to produce deformed images according to a given deformation model
# among: rdf, gbd2, gbd3, cnb, cpab, fbm
def apply_deformation(deformation, sub_dir, subset, N, height, width, params=None, baseline=False):

	save_dir = sub_dir + '/' + deformation
	image_dir = '../Data/' + sub_dir + '/%s/images' % subset
	gt_dir    = '../Data/' + sub_dir + '/%s/gts' % subset

	if deformation == 'fbm':
		import FBM_aug as fbm
		fbm.fbm(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
	elif deformation == 'gdb3':
		import GBD_aug as gbd
		gbd.gbd(image_dir, gt_dir, N, save_dir, subset, 3, params[0], params[1], height, width)
	elif deformation == 'gdb2':
		import GBD_aug as gbd
		gbd.gbd(image_dir, gt_dir, N, save_dir, subset, 2, params[0], params[1], height, width)
	elif deformation == 'rdf':
		import RDF_aug as rdf
		rdf.rdf(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
	elif deformation == 'cpab':
		import CPAB_aug as cpab
		cpab.cpab(image_dir, gt_dir, N, save_dir, subset, params[0], params[1])
	elif deformation == 'cnb':
		import CNB_aug as cnb
		nuclei_results = '../Data/' + sub_dir + '/nuclei_segmentation_results'
		cnb.cnb_mls(image_dir, gt_dir, nuclei_results, N, params[0], save_dir, subset)
	else:
		print('Deformation not known or implemented yet.')



# PARAMS

# Size of the patches to deform and used for segmentation
height = 256
width  = 256

# In the following, the patches to be deformed are supposed to be
# crops extracted from 512*512 patches. This can be skipped
# according to user's needs.

# Number of 512*512 patches to select for each subset. 
# To this number we add the same number of negative patches (see tree below). 
N_test  = 50                                                    
N_train = 20 # 10, 20, 100, 300, 600
N_val   = 10 # 5, 10, 30, 100, 200 

# Directory containing the 512*512 patches among which height*width
# patches are cropped
orig_dir = 'patches'

# Original tree of our data:
# Data
# ├── patches
# │   ├── test
# │   │   ├── gts
# │   │   │   ├── background
# │   │   │   ├── glomeruli
# │   │   │   └── negative
# │   │   └── images
# │   │       ├── background
# │   │       ├── glomeruli
# │   │       └── negative
# │   ├── train
# │   │   ├── gts
# │   │   │   ├── background
# │   │   │   ├── glomeruli
# │   │   │   └── negative
# │   │   └── images
# │   │       ├── background
# │   │       ├── glomeruli
# │   │       └── negative
# │   └── validation
# │       ├── gts
# │       │   ├── background
# │       │   ├── glomeruli
# │       │   └── negative
# │       └── images
# │           ├── background
# │           ├── glomeruli
# │           └── negative

# The function prepare_patches selects the number of images according to the given
# parameters in each folder, then performs nb_crops croppings (with nb_crops=5).
# The resulting tree looks like this:
# Data
# ├── patches_cropped_N_train
# │   ├── test
# │   │   ├── gts
# │   │   └── images
# │   ├── train
# │   │   ├── gts
# │   │   └── images
# │   └── validation
# │       ├── gts
# │       └── images

# The function prepare_patches returns the name of the folder that contains
# the patches to be deformed and used for training of the segmentation model.

print("Prepare patches")
save_dir = prepare_patches([N_train, N_val, N_test], ['train', 'validation', 'test'], 5, orig_dir, height, width)

# If prepare_patches is not used, a value for save_dir has to be provided.
#save_dir = 'patches_20'

# Number of deformations for each patch.
nb_deform = 10


# Performs all deformation with the parameters given above.
# Comment/Uncomment according to user's needs.

print("Augment GBD3")
# Grid Search. Change values according to user's needs.
param_n = [3]#[3, 5, 10]
param_sigma = [20]#[5, 10, 20, 50]
for n in param_n:
	for sigma in param_sigma:
		start = time.time()
		apply_deformation('gdb3', save_dir, 'train', nb_deform, height, width, [n, sigma])
		end = time.time()
		print("Deformation time for dataset:", '{:.4f} s'.format(end-start))


print("Augment GBD2")
# Grid Search. Change values according to user's needs.
param_n = [3]#[3, 5, 10]
param_sigma = [20]#[5, 10, 20, 50]
for n in param_n:
	for sigma in param_sigma:
		start = time.time()
		apply_deformation('gdb2', save_dir, 'train', nb_deform, height, width, [n, sigma])
		end = time.time()
		print("Deformation time for dataset:", '{:.4f} s'.format(end-start))


print("Augment RDF")
# Grid Search. Change values according to user's needs.
param_alpha = [50]#[50, 100, 200, 400]
param_sigma2 = [5]#[5, 10, 20]
for alpha in param_alpha:
	for sigma in param_sigma2:
		start = time.time()
		apply_deformation('rdf', save_dir, 'train', nb_deform, height, width, [alpha, sigma])
		end = time.time()
		print("Deformation time for dataset:", '{:.4f} s'.format(end-start))

		
print("Augment FBM")
# Grid Search. Change values according to user's needs.
param_w = [0.1]#[0.1, 0.35, 0.7, 1]
param_s = [1]#[1, 2, 4]
for w in param_w:
	for s in param_s:
		start = time.time()
		apply_deformation('fbm', save_dir, 'train', nb_deform, height, width, [w, s])
		end = time.time()
		print("Deformation time for dataset:", '{:.4f} s'.format(end-start))

		
print("Augment CPAB")		
# Grid Search. Change values according to user's needs.
param_n = [3]#[3, 5, 10, 15]
param_var = [0.5]#[0.5, 1, 2, 5, 10]
for n in param_n:
		for var in param_var:
				start = time.time()
				apply_deformation('cpab', save_dir, 'train', nb_deform, height, width, [n, var])
				end = time.time()
				print("Deformation time for dataset:", '{:.4f} s'.format(end-start))

		
print("Augment CNB")
# Grid Search. Change values according to user's needs.
param_sigma = [5]#[5, 10, 15, 30, 100]
for sigma in param_sigma:
	start = time.time()
	apply_deformation('cnb', save_dir, 'train', nb_deform, height, width, [sigma])
	end = time.time()
	print("Deformation time for dataset:", '{:.4f} s'.format(end-start))

