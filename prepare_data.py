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
import math
import time
import random

import PIL
from PIL import Image
from skimage import img_as_ubyte

import tensorflow as tf
import tensorflow_addons as tfa
tf.executing_eagerly()

from joblib import Parallel, delayed
import multiprocessing

import noise

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io, img_as_float, img_as_ubyte

MAX_NOISE_RAND = 1024

def fetch_file_names(file_dir):
    file_names = glob('%s/*.png' % file_dir)
    return np.array(file_names)

def read_image(file_name, channels):

	file_name_noext = os.path.splitext(file_name)[0].split('/')[-1]

	img_file = tf.io.read_file(file_name)
	img = tf.image.decode_png(img_file, channels=channels, dtype=tf.uint8)
	img = tf.image.convert_image_dtype(img, dtype=tf.float32)
	# Add batch dimension
	img = tf.expand_dims(img, axis=0)

	return img, file_name_noext           

def prepare_crop(N, subsets, orig_dir, height, width, random=False):

    for i in range(len(subsets)):

        subset = subsets[i]
        n = N[i]

        rd = ""
        if random:
            rd = "rd"

        save_dir = '../Data/' + orig_dir + '_' + str(N[0]) + "rd" + '/' + subset       

        if not os.path.exists('%s/images/' % save_dir):
            os.makedirs('%s/images/' % save_dir)
        if not os.path.exists('%s/gts/' % save_dir):
            os.makedirs('%s/gts/' % save_dir)

        images_glomeruli = fetch_file_names('../Data/' + orig_dir + '/%s/images/glomeruli' % subset)
        images_negative  = fetch_file_names('../Data/' + orig_dir + '/%s/images/negative' % subset)
        gts_glomeruli    = fetch_file_names('../Data/' + orig_dir + '/%s/gts/glomeruli' % subset)
        gts_negative     = fetch_file_names('../Data/' + orig_dir + '/%s/gts/negative' % subset)

        for i in range(n):

            if random :
                idx = random.randint(0,len(images_glomeruli)-1)
            else:
                idx = i

            print('cropping image %d/%d' % (i+1, n))
            
            file_image_glomeruli = images_glomeruli[idx]
            file_gt_glomeruli = gts_glomeruli[idx]

            image_glomeruli = PIL.Image.open(file_image_glomeruli).copy()
            gt_glomeruli = PIL.Image.open(file_gt_glomeruli).copy()
            
            file_image_negative = images_negative[idx]
            file_gt_negative = gts_negative[idx]

            image_negative = PIL.Image.open(file_image_negative)
            gt_negative = PIL.Image.open(file_gt_negative)

            file_image_glomeruli_noext = os.path.splitext(file_image_glomeruli)[0].split('/')[-1]
            file_gt_glomeruli_noext    = os.path.splitext(file_gt_glomeruli)[0].split('/')[-1]
            file_image_negative_noext  = os.path.splitext(file_image_negative)[0].split('/')[-1]
            file_gt_negative_noext     = os.path.splitext(file_gt_negative)[0].split('/')[-1]

            for j in range(5):
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

            
def apply_deformation(deformation, sub_dir, subset, N, height, width, params=None, baseline=False):

    save_dir = sub_dir + '/' + deformation
    image_dir = '../Data/' + sub_dir + '/%s/images' % subset
    gt_dir    = '../Data/' + sub_dir + '/%s/gts' % subset

    if deformation == 'fbm':
        fbm(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
    elif deformation == 'gdb3':
        gdb(image_dir, gt_dir, N, save_dir, subset, 3, params[0], params[1], height, width)
    elif deformation == 'gdb2':
        gdb(image_dir, gt_dir, N, save_dir, subset, 2, params[0], params[1], height, width)
    elif deformation == 'rdf':
        rdf(image_dir, gt_dir, N, save_dir, subset, params[0], params[1], height, width)
    elif deformation == 'foldover':
        foldover(image_dir, gt_dir, N, save_dir, subset, height, width)
    else:
        print('Deformation not known or implemented yet.')

def phi(x):
	return noise.NoiseP[x % MAX_NOISE_RAND]

def inoiseG(ix, iy, noise_tab):
	index = (phi(ix) + 3 * phi(iy)) % MAX_NOISE_RAND

	theta = (noise_tab[index] + 1.0) * math.pi

	G = [math.cos(theta), math.sin(theta)] 
	G = G / np.sqrt(G[0]*G[0] + G[1]*G[1])

	return G

def mix(x, y, a):
	return x * (1 - a) + y * a

def dot(x1, y1, x2, y2):
	return x1 * x2 + y1 * y2

def cnoise2DG(x, y, noise_tab):
	ix = np.floor(x).astype('int32')
	x -= ix
	iy = np.floor(y).astype('int32')
	y -= iy
	
	sx = (x * x * (3.0 - 2.0 * x))
	sy = (y * y * (3.0 - 2.0 * y))
	
	vy0 = inoiseG(ix, iy, noise_tab)
	vy1 = inoiseG(ix, iy + 1, noise_tab)

	vx0 = mix( dot(vy0[0], vy0[1], x - 0.0, y - 0.0),
	           dot(vy1[0], vy1[1], x - 0.0, y - 1.0), sy )
	
	vy0 = inoiseG(ix + 1, iy, noise_tab)
	vy1 = inoiseG(ix + 1, iy + 1, noise_tab)
	vx1 = mix( dot(vy0[0], vy0[1], x - 1.0, y - 0.0),
	           dot(vy1[0], vy1[1], x - 1.0, y - 1.0), sy )
	
	rt = mix( vx0, vx1, sx )
	
	return rt

def fBMGd(distortamp, distortfact, distortfreq, x, y, noise_tab): # distortamp = w, distortfact = 0.5, distorfreq = 2*s
	cn1 = distortamp * cnoise2DG( distortfreq*x + 2.0, distortfreq*y, noise_tab )
	cn2 = distortamp * distortfact * cnoise2DG( distortfreq*x*2.0 + 2.0, distortfreq*y*2.0, noise_tab)
	cn3 = distortamp * distortfact * distortfact * cnoise2DG( distortfreq*x*4.0 + 2.0, distortfreq*y*4.0, noise_tab )
	cn4 = distortamp * distortfact * distortfact * distortfact * cnoise2DG( distortfreq*x*8.0 + 2.0, distortfreq*y*8.0, noise_tab )
	dx =  (cn1 + cn2/2.0 + cn3/4.0 + cn4/8.0) / distortfreq
	
	cn1 = distortamp * cnoise2DG( distortfreq*x, distortfreq*y + 5.0, noise_tab  )
	cn2 = distortamp * distortfact * cnoise2DG( distortfreq*x*2.0, distortfreq*y*2.0 + 5.0, noise_tab )
	cn3 = distortamp * distortfact * distortfact * cnoise2DG( distortfreq*x*4.0, distortfreq*y*4.0 + 5.0, noise_tab )
	cn4 = distortamp * distortfact * distortfact * distortfact * cnoise2DG( distortfreq*x*8.0, distortfreq*y*8.0 + 5.0, noise_tab )
	dy =  (cn1 + cn2/2.0 + cn3/4.0 + cn4/8.0) / distortfreq
	
	return (dx, dy)

def processInput(k, height, width, distortamp, distortfact, distortfreq, slice_size_tab, slice_size, noise_tab):
	fBM_tab = np.zeros((slice_size_tab[k], width, 2), dtype = float)
	for i in range(slice_size_tab[k]): # Lines
		for j in range(width): # Columns
			ii = (i +  k * slice_size) / height
			jj = j / width
			fBM_djj, fBM_dii = fBMGd(distortamp, distortfact, distortfreq, jj, ii, noise_tab)
			fBM_tab[i, j][0] = fBM_dii * height
			fBM_tab[i, j][1] = fBM_djj * width
	return fBM_tab


def fbm(file_dir, mask_dir, essais, save_dir, subset, w, s, height, width):

	ext = '_1T_%.3f_%.3f' % (w, s)

	pad_size = 128

	if not os.path.exists('../Data/%s/%s/images/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/images/' % (save_dir+ext, subset))
	if not os.path.exists('../Data/%s/%s/gts/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/gts/' % (save_dir+ext, subset))

	# distortamp = w, distortfact = 0.5, distorfreq = 2*s
			
	distortamp  = w    
	distortfreq = s    
	distortfact = 0.5    

	file_names = fetch_file_names(file_dir)
	mask_names = fetch_file_names(mask_dir)

	num_cores = multiprocessing.cpu_count()
	inputs = range(num_cores)

	for f in range(len(file_names)):

		print("fbm ", f+1, "/", len(file_names))

		img, file_name_noext = read_image(file_names[f], 3)
		gt, gt_name_noext = read_image(mask_names[f], 1)

		height_orig = tf.shape(img)[1].numpy()
		width_orig  = tf.shape(img)[2].numpy()

		# Add padding
		paddings = tf.constant([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		img = tf.pad(img[0,:,:,:], paddings, "REFLECT") # [0,:,:,:]
		gt = tf.pad(gt[0,:,:,:], paddings, "REFLECT") # [0,:,:,:]

		# Add batch dimension
		img = tf.expand_dims(img, axis=0)
		gt = tf.expand_dims(gt, axis=0)

		height = tf.shape(img)[1].numpy()
		width  = tf.shape(img)[2].numpy()

		slice_size = height // num_cores
		slice_size_remaining = height % num_cores
		slice_size_tab = np.zeros((num_cores,), dtype=int)
		if slice_size_remaining:
			for k in range(num_cores - 1):
				slice_size_tab[k] = slice_size
			slice_size_tab[num_cores - 1] = slice_size + slice_size_remaining
		else:
			for k in range(num_cores):
				slice_size_tab[k] = slice_size

		for n in range(essais):

			print(n)

			noise_tab = np.random.rand(MAX_NOISE_RAND)
			noise_tab *= 2
			noise_tab -= 1

			fBM_tab_slices = Parallel(n_jobs=num_cores)(delayed(processInput)(i, height, width, 
																		distortamp, distortfact, distortfreq, 
																		slice_size_tab, slice_size, noise_tab) for i in inputs)
			fBM_tab = np.concatenate(fBM_tab_slices)

			fBM_tab = tf.convert_to_tensor(fBM_tab, dtype=tf.float32)
			#fBM_tab = tf.pad(fBM_tab, paddings, "REFLECT")
			fBM_tab = tf.expand_dims(fBM_tab, axis=0)
			
			print(tf.shape(img))
			print(tf.shape(fBM_tab))
			# Warp image
			dense_img_warp = tfa.image.dense_image_warp(img, fBM_tab)
			# Warp gt
			dense_gt_warp = tfa.image.dense_image_warp(gt, fBM_tab)

			# Remove padding
			dense_img_warp = tf.image.crop_to_bounding_box(dense_img_warp, pad_size, pad_size, height_orig, width_orig)
			dense_gt_warp = tf.image.crop_to_bounding_box(dense_gt_warp, pad_size, pad_size, height_orig, width_orig)

			# Remove batch dimension
			dense_img_warp = tf.squeeze(dense_img_warp, 0)
			dense_gt_warp = tf.squeeze(dense_gt_warp, 0)		
				
			# Write images
			dense_img_warp = tf.image.convert_image_dtype(dense_img_warp, dtype=tf.uint8)
			dense_img_warp_png = tf.io.encode_png(dense_img_warp)
			tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/images/'+file_name_noext+'_'+str(n)+'.png', dense_img_warp_png)

			dense_gt_warp = tf.image.convert_image_dtype(dense_gt_warp, dtype=tf.uint8)
			dense_gt_warp_png = tf.io.encode_png(dense_gt_warp)
			tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(n)+'.png', dense_gt_warp_png)

def gdb(file_dir, mask_dir, essais, save_dir, subset, order, n, sigma, height, width):

	ext = '_%.3f_%.3f' % (n, sigma)

	pad_size = 128

	if not os.path.exists('../Data/%s/%s/images/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/images/' % (save_dir+ext, subset))
	if not os.path.exists('../Data/%s/%s/gts/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/gts/' % (save_dir+ext, subset))	

	file_names = fetch_file_names(file_dir)
	mask_names = fetch_file_names(mask_dir)

	for f in range(len(file_names)):

		print("gdb ", f+1, "/", len(file_names))

		with tf.device('GPU:0'):

			img, file_name_noext = read_image(file_names[f], 3)
			gt, gt_name_noext = read_image(mask_names[f], 1)

			# Add padding
			paddings = tf.constant([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])
			img = tf.pad(img[0,:,:,:], paddings, "REFLECT")
			gt = tf.pad(gt[0,:,:,:], paddings, "REFLECT")

			for e in range(essais):

				displacement = tf.random.normal([n*n, 2], 0.0, sigma, dtype=tf.float32)
	
				x = tf.linspace(0.0 + pad_size, height * 1.0 + pad_size, n)
				y = tf.linspace(0.0 + pad_size, width * 1.0 + pad_size, n)
				X, Y = tf.meshgrid(x, y)

				source = tf.stack([Y, X], axis=2)
				source = tf.reshape(source, [n*n, 2])
				dest = tf.add(source, displacement)
				
	
				source = tf.expand_dims(source, axis=0)
				dest = tf.expand_dims(dest, axis=0)
	
				# Warp image
				# https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp
				dense_img_warp, flow_field = tfa.image.sparse_image_warp(img, source, dest,
	                                                         		num_boundary_points=0,
															 		interpolation_order=order)
				dense_gt_warp, flow_field = tfa.image.sparse_image_warp(gt, source, dest,
	                                                         		num_boundary_points=0,
															 		interpolation_order=order)


				# Remove padding
				dense_img_warp = tf.image.crop_to_bounding_box(dense_img_warp, pad_size, pad_size, height, width)
				dense_gt_warp = tf.image.crop_to_bounding_box(dense_gt_warp, pad_size, pad_size, height, width)

				# Remove batch dimension
				#dense_img_warp = tf.squeeze(dense_img_warp, 0)
				#dense_gt_warp = tf.squeeze(dense_gt_warp, 0)

				# Write images
				dense_img_warp = tf.image.convert_image_dtype(dense_img_warp, dtype=tf.uint8)
				dense_img_warp_png = tf.io.encode_png(dense_img_warp)
				tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/images/'+file_name_noext+'_'+str(e)+'.png', dense_img_warp_png)

				dense_gt_warp = tf.image.convert_image_dtype(dense_gt_warp, dtype=tf.uint8)
				dense_gt_warp_png = tf.io.encode_png(dense_gt_warp)
				tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(e)+'.png', dense_gt_warp_png)

def foldover(file_dir, mask_dir, essais, save_dir, subset, height, width):

	ext = ''

	pad_size = 128

	n=3
	sigma=50

	if not os.path.exists('../Data/%s/%s/images/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/images/' % (save_dir+ext, subset))
	if not os.path.exists('../Data/%s/%s/gts/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/gts/' % (save_dir+ext, subset))	

	file_names = fetch_file_names(file_dir)
	mask_names = fetch_file_names(mask_dir)

	for f in range(len(file_names)):

		print("gdb ", f+1, "/", len(file_names))

		with tf.device('GPU:0'):

			img, file_name_noext = read_image(file_names[f], 3)
			gt, gt_name_noext = read_image(mask_names[f], 1)

			# Add padding
			paddings = tf.constant([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])
			img = tf.pad(img[0,:,:,:], paddings, "REFLECT")
			gt = tf.pad(gt[0,:,:,:], paddings, "REFLECT")

			e = 0
			while e != essais:

				displacement = tf.random.normal([n*n, 2], 0.0, sigma, dtype=tf.float32)
	
				x = tf.linspace(0.0 + pad_size, height * 1.0 + pad_size, n)
				y = tf.linspace(0.0 + pad_size, width * 1.0 + pad_size, n)
				X, Y = tf.meshgrid(x, y)

				source = tf.stack([Y, X], axis=2)
				source = tf.reshape(source, [n*n, 2])
				dest = tf.add(source, displacement)
				
	
				source = tf.expand_dims(source, axis=0)
				dest = tf.expand_dims(dest, axis=0)

				fold = foldover_calc(source, displacement, height, width, pad_size)
				print(fold)
				if fold < 5:
	
					# Warp image
					# https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp
					dense_img_warp, flow_field = tfa.image.sparse_image_warp(img, source, dest,
																		num_boundary_points=0,
																		interpolation_order=3)
					dense_gt_warp, flow_field = tfa.image.sparse_image_warp(gt, source, dest,
																		num_boundary_points=0,
																		interpolation_order=3)


					# Remove padding
					dense_img_warp = tf.image.crop_to_bounding_box(dense_img_warp, pad_size, pad_size, height, width)
					dense_gt_warp = tf.image.crop_to_bounding_box(dense_gt_warp, pad_size, pad_size, height, width)

					# Remove batch dimension
					#dense_img_warp = tf.squeeze(dense_img_warp, 0)
					#dense_gt_warp = tf.squeeze(dense_gt_warp, 0)



					# Write images
					dense_img_warp = tf.image.convert_image_dtype(dense_img_warp, dtype=tf.uint8)
					dense_img_warp_png = tf.io.encode_png(dense_img_warp)
					tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/images/'+file_name_noext+'_'+str(e)+'.png', dense_img_warp_png)

					dense_gt_warp = tf.image.convert_image_dtype(dense_gt_warp, dtype=tf.uint8)
					dense_gt_warp_png = tf.io.encode_png(dense_gt_warp)
					tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(e)+'.png', dense_gt_warp_png)

					e = e+1

def foldover_calc(source, displacement, height, width, pad_size):

	im_size_pad = [height + pad_size, width + pad_size]
	img = np.zeros((im_size_pad[0] * im_size_pad[1], 2))
	for i in range(im_size_pad[0]):
		for j in range(im_size_pad[1]):
			img[i*im_size_pad[1]+j][0] = j
			img[i*im_size_pad[1]+j][1] = i
	img = tf.convert_to_tensor(img, dtype=tf.float32)
	img = tf.expand_dims(img, axis=0)

	displacement = tf.expand_dims(displacement, axis=0)

	# Apply spline interpolation to compute displacements
	img_displacement = tfa.image.interpolate_spline(source, displacement,
													img, order=3)
				
	dimg = tf.add(img, img_displacement)
	dimg = tf.reshape(dimg, (1, im_size_pad[0], im_size_pad[1], 2))
				
	# Remove padding
	dimg = tf.image.crop_to_bounding_box(dimg, pad_size, pad_size, height, width)


	d1y, d1x = tf.image.image_gradients(tf.expand_dims(dimg[:,:,:,0], axis=-1))
	d2y, d2x = tf.image.image_gradients(tf.expand_dims(dimg[:,:,:,1], axis=-1))
	detJ = tf.math.multiply(d1x[0,:,:,0], d2y[0,:,:,0]) - tf.math.multiply(d1y[0,:,:,0], d2x[0,:,:,0])
	detJ = detJ.numpy()
	detJ_np = np.zeros_like(detJ)
	detJ_np[detJ < 1e-6] = 1.0
	# Exclude the last column and the last line
	detJ_np[height-1:,:] = 0.0
	detJ_np[:,width-1:] = 0.0
	count = np.sum(detJ_np)
	foldover = (count * 100.0) / (height*width)

	return foldover


def rdf(file_dir, mask_dir, essais, save_dir, subset, alpha, sigma, height, width):

	"""Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """


	ext = '_%.3f_%.3f' % (alpha, sigma)
	pad_size=128

	if not os.path.exists('../Data/%s/%s/images/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/images/' % (save_dir+ext, subset))
	if not os.path.exists('../Data/%s/%s/gts/' % (save_dir+ext, subset)):
		os.makedirs('../Data/%s/%s/gts/' % (save_dir+ext, subset))	

	file_names = fetch_file_names(file_dir)
	mask_names = fetch_file_names(mask_dir)

	for f in range(len(file_names)):

		print("sim ", f+1, "/", len(file_names))
		with tf.device('GPU:0'):

			img, file_name_noext = read_image(file_names[f], 3)
			gt, gt_name_noext = read_image(mask_names[f], 1)

			n = tf.shape(img)[1]

			for e in range(essais):

				radius = 4*sigma + 0.5
				fs = 2*int(radius)+1

				displacement = tf.random.uniform([n, n, 2], -1, 1, dtype=tf.float32)
				displacement = tfa.image.gaussian_filter2d(displacement, filter_shape=fs, sigma=sigma, padding='REFLECT') * alpha
		
				x = tf.linspace(0.0, height * 1.0 -1, n)
				y = tf.linspace(0.0, width * 1.0 -1, n)
				X, Y = tf.meshgrid(x, y)

				source = tf.stack([Y, X], axis=2)
				#source = tf.reshape(source, [n*n, 2])
				dest = tf.add(source, displacement)

				dest = tf.expand_dims(dest, axis=0)
				displacement = tf.expand_dims(displacement, axis=0)

				# Warp image
				# https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
				dense_img_warp = tfa.image.dense_image_warp(img, displacement)
				dense_gt_warp = tfa.image.dense_image_warp(gt, displacement)

				# Remove batch dimension
				dense_img_warp = tf.squeeze(dense_img_warp, 0)
				dense_gt_warp = tf.squeeze(dense_gt_warp, 0)

				# Write images
				dense_img_warp = tf.image.convert_image_dtype(dense_img_warp, dtype=tf.uint8)
				dense_img_warp_png = tf.io.encode_png(dense_img_warp)
				tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/images/'+file_name_noext+'_'+str(e)+'.png', dense_img_warp_png)

				dense_gt_warp = tf.image.convert_image_dtype(dense_gt_warp, dtype=tf.uint8)
				dense_gt_warp_png = tf.io.encode_png(dense_gt_warp)
				tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(e)+'.png', dense_gt_warp_png)

  
# PARAMS

height = 256
width  = 256

N_test  = 50

# 1er sous ensemble                                                     
N_train = 20 # 10, 20, 100, 300, 600
N_val   = 10 # 5, 10, 30, 100, 200 

#orig_dir = 'selected_patches'
orig_dir = 'patches'

print("prepare crop")
#save_dir = prepare_crop([N_train, N_val, N_test], ['train', 'validation', 'test'], orig_dir, height, width)
#save_dir = prepare_crop([N_train, N_val], ['train', 'validation'], orig_dir, height, width)
save_dir = 'patches_cropped_20'

essais = 10

# Grid Search
param_w = [0.1, 0.35, 0.7, 1]
param_s = [1, 2, 4]

param_n = [3, 5, 10]
param_sigma = [5, 10, 20, 50]

param_alpha = [50, 100, 200, 400]
param_sigma2 = [5, 10, 20]



print("augment gdb3")
for n in param_n:
	for sigma in param_sigma:
		start = time.time()
		apply_deformation('gdb3', save_dir, 'train', essais, height, width, [n, sigma])
		end = time.time()
		print("Dataset warping time:", '{:.4f} s'.format(end-start))


print("augment gdb2")
for n in param_n:
	for sigma in param_sigma:
		start = time.time()
		apply_deformation('gdb2', save_dir, 'train', essais, height, width, [n, sigma])
		end = time.time()
		print("Dataset warping time:", '{:.4f} s'.format(end-start))


print("augment rdf")
for alpha in param_alpha:
	for sigma in param_sigma2:
		start = time.time()
		apply_deformation('rdf', save_dir, 'train', essais, height, width, [alpha, sigma])
		end = time.time()
		print("Dataset warping time:", '{:.4f} s'.format(end-start))

print("augment fbm")
for w in param_w:
	for s in param_s:
		start = time.time()
		apply_deformation('fbm', save_dir, 'train', essais, height, width, [w, s])
		end = time.time()
		print("Dataset warping time:", '{:.4f} s'.format(end-start))
