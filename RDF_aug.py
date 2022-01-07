


def rdf(file_dir, mask_dir, essais, save_dir, subset, alpha, sigma, height, width):

	"""Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """


	ext = '_%.3f_%.3f' % (alpha, sigma)
	pad_size = int(max(width, height) / 2)

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