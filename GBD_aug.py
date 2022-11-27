


def gbd(file_dir, mask_dir, essais, save_dir, subset, order, n, sigma, height, width):

	ext = '_%.3f_%.3f' % (n, sigma)

	pad_size = int(max(width, height) / 2)

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
				tf.io.write_file('../Data/'+save_dir+ext+'/'+subset+'/gts/'+file_name_noext+'_'+str(e)+'.png', 255*dense_gt_warp_png)




def gdb3_no_foldover(file_dir, mask_dir, essais, save_dir, subset, height, width, rate, n=3, sigma=50):
	
	# This fonction operates the same as gdb, but discard the deformation fields with a foldover rate > rate

	ext = ''

	pad_size = int(max(width, height) / 2)

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
				if fold < rate:
	
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