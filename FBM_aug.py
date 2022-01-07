MAX_NOISE_RAND = 1024

from joblib import Parallel, delayed
import multiprocessing

import noise

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

	pad_size = int(max(width, height) / 2)

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