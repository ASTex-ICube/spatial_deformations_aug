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

# Based on the paper:
# S. Schaefer, T. McPhail, J. Warren, Image deformation using moving
# least squares, ACM Trans. Graph. 25 (3) (2006) 533–540. doi:10.1145/
# 1141911.1141920. URL https://doi.org/10.1145/1141911.1141920
# Based on the implementation by Jarvis73
# https://github.com/Jarvis73/Moving-Least-Squares

import os
import sys
import numpy as np 
import time
from skimage import io, img_as_float, img_as_ubyte, color, filters, exposure, morphology, draw, transform
from scipy.ndimage.interpolation import map_coordinates
#from cell_nuclei_segmentation import nuclei_DS
#import elasticdeform
from scipy.interpolate import Rbf
from glob import glob

from scipy.spatial.distance import squareform, pdist

#----- MLS deformation functions -----

np.seterr(divide='ignore', invalid='ignore')

def mls_rigid_deformation(image, mask, p, q, alpha=1.0):
	''' Rigid deformation
	### Params:
		* image - ndarray: original image
        * mask - ndarray: mask
		* p - ndarray: an array with size [n, 2], original control points
		* q - ndarray: an array with size [n, 2], final control points
		* alpha - float: parameter used by weights
	### Return:
		A deformed image.
	'''
	
	height = image.shape[0]
	width = image.shape[1]
	# Change (x, y) to (row, col)
	#q = q[:, [1, 0]]
	#p = p[:, [1, 0]]

	# Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
	p, q = q, p

	# Make grids on the original image
	gridX = np.linspace(0, width, num=width, endpoint=False)
	gridY = np.linspace(0, height, num=height, endpoint=False)
	vy, vx = np.meshgrid(gridX, gridY)
	grow = vx.shape[0]  # grid rows
	gcol = vx.shape[1]  # grid cols
	ctrls = p.shape[0]  # control points

	# Compute
	reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
	reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
	
	w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
	sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
	pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
	phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
	reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
	reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
	neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
	neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
	reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
	mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)         # [ctrls, 2, 2, grow, gcol]
	vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
	reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
	neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
	neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
	reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
	mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
	reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
	A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2), 
					   reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

	# Calculate q
	reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
	qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
	qhat = reshaped_q - qstar                                                           # [2, grow, gcol]
	reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

	# Get final image transformer -- 3-D array
	temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
	reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
	norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)           # [1, grow, gcol]
	norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                         # [1, grow, gcol]
	transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar  + qstar            # [2, grow, gcol]
	
	indices = np.reshape(transformers[0], (-1, 1)), np.reshape(transformers[1], (-1, 1))
	
	R = image[:,:,0]
	G = image[:,:,1]
	B = image[:,:,2]
	R = map_coordinates(R, indices, order=3, mode='reflect').reshape(height, width)
	G = map_coordinates(G, indices, order=3, mode='reflect').reshape(height, width)
	B = map_coordinates(B, indices, order=3, mode='reflect').reshape(height, width)
	transformed_image1 = np.stack((R, G, B), axis = 2)
	
    transformed_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(height, width)
    
    return transformed_image1, mask
	

#----- Image loading and deformation -----

def fetch_file_names(file_dir):

	file_names = glob('%s/*.png' % file_dir)

	return np.array(file_names)

def mls(image_dir, gt_dir, nuclei_results, N, n, sigma, save_dir, subset):

	ext = '_%.3f_%.3f' % (n, sigma)
      
	file_names = fetch_file_names(image_dir)
	mask_names = fetch_file_names(gt_dir)

	if not os.path.exists('../Data/%s/%s/images/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/images/' % (save_dir+ext, subset))
	if not os.path.exists('../Data/%s/%s/gts/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/gts/' % (save_dir+ext, subset))

	for f in range(len(file_names)):

		print("mls ", f, "/", len(file_names))

		file_name = file_names[f]
		file_name_noext = os.path.splitext(file_name)[0].split('/')[-1]

		mask_name = mask_names[f]
		mask_name_noext = os.path.splitext(mask_name)[0].split('/')[-1]

		#print(nuclei_results + '/' + file_name_noext + '.txt')
		#print(file_name_noext)

		file_name_nuclei_centers = nuclei_results + '/' + file_name_noext + '_nuclei_centers.txt'

		# file_name_nuclei_centers
	
		# Load nuclei centers
		center_array = np.loadtxt(file_name_nuclei_centers, dtype=int)
		center_array = center_array[:, [1, 0]]
		# Remove duplicates
		center_array = np.unique(center_array, axis=0)
                
		# MLS as-rigid-as-possible deformation
                
		input_image = io.imread(file_name)
		input_image = img_as_float(input_image)
                
		pad_size = 0
		shape = input_image.shape
                
		# Pad the edges of the input image with symmetric content
		input_image = np.pad(input_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="symmetric")
		center_array += pad_size

		input_mask = io.imread(mask_name, as_gray=True)
		input_mask = img_as_float(input_mask)
		
		# Pad the edges of the input image with symmetric content
		input_mask = np.pad(input_mask, ((pad_size, pad_size), (pad_size, pad_size)), mode="symmetric")
		
		max_dist_img = math.sqrt(input_image.shape[0]*input_image.shape[0] + input_image.shape[1]*input_image.shape[1])
		# Number of detected cell nuclei centers
		n = center_array.shape[0]
		
		for i in range(N):
			
			dist = squareform(pdist(center_array))
			cov = np.exp(-(dist / max_dist_img))
			sample = np.random.multivariate_normal(np.zeros(n), cov, size=2) * sigma
			sample = np.transpose(sample)
			p = center_array
			q = p + sample
			
			'''
			displacement = np.random.randn(n*n, 2) * sigma
			rbfi = Rbf(x, y, displacement, function='thin_plate', mode='N-D')
			p = center_array
			q = []
			for k in range(p.shape[0]):
				px = (p[k][1] / input_image.shape[1]) * 2 - 1
				py = (p[k][0] / input_image.shape[0]) * 2 - 1
				d = rbfi(px, py)
				q.append(p[k] + d)
			q = np.array(q)
			'''
                        
			output_image, output_mask = mls_rigid_deformation(input_image, p, q, alpha=2.0)
			# Remove the padding
			output_image = output_image[pad_size:shape[0]+pad_size, pad_size:shape[1]+pad_size]
			output_image = np.clip(output_image, 0, 1)
			output_image_ubyte = img_as_ubyte(output_image)
			io.imsave('../Data/'+save_dir+ext+'/'+subset+'/images/'+file_name_noext+'_'+str(i)+'.png', output_image_ubyte)
			
			# Remove the padding
			output_mask = output_mask[pad_size:shape[0]+pad_size, pad_size:shape[1]+pad_size]
			output_mask = np.clip(output_mask, 0, 1)
			output_mask[output_mask > 0.0] = 1.0
			output_mask_ubyte = img_as_ubyte(output_mask)
			io.imsave('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(i)+'.png', output_mask_ubyte, check_contrast=False)
	
def apply_deformation(deformation, sub_dir, subset, N, height, width, params=None, baseline=False):

	save_dir = sub_dir + '/' + deformation
	image_dir = '../Data/' + sub_dir + '/%s/images' % subset
	gt_dir    = '../Data/' + sub_dir + '/%s/gts' % subset

	if deformation == 'fbm':
		fbm(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
	elif deformation == 'elt':
		elt(image_dir, gt_dir, N, save_dir, subset, 3, params[0], params[1], height, width)
	elif deformation == 'tps':
		elt(image_dir, gt_dir, N, save_dir, subset, 2, params[0], params[1], height, width)
	elif deformation == 'sim':
		sim(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
	elif deformation == 'mls':
		nuclei_results = '../Data/' + sub_dir + '/nuclei_segmentation_results'
		mls(image_dir, gt_dir, nuclei_results, N, params[0], params[1], save_dir, subset)
	else:
		print('Deformation not known or implemented yet.')

height = 256
width  = 256
essais = 10

param_n = [3, 5, 10]
param_sigma = [5, 10, 20, 50]

sub_dir = 'patches_cropped_20'

print("augment mls")
for n in param_n:
	for sigma in param_sigma:
		start = time.time()
		apply_deformation('mls', sub_dir, 'train', essais, height, width, [n, sigma])
		end = time.time()
		print("Dataset warping time:", '{:.4f} s'.format(end-start))