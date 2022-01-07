# -*- coding: utf-8 -*-

#%%
from libcpab import Cpab
import numpy as np
import argparse
import os
import sys
import torch

from skimage import io, img_as_float, img_as_ubyte
from glob import glob
import time


# https://github.com/SkafteNicki/libcpab

#%%
def fetch_file_names(file_dir):
    file_names = glob('%s/*.png' % file_dir)
    return np.array(file_names)

def cpab_transfo(file_names, mask_names, N, save_dir, subset, n, var):
	
        args_backend = 'pytorch' # choices=['numpy', 'tensorflow', 'pytorch'],
        args_device = 'gpu' # choices=['cpu', 'gpu']

        import tensorflow as tf
        tf.config.set_soft_device_placement(True)
                

        # Create transformer class
        T = Cpab([n, n], backend=args_backend, device=args_device, 
                         zero_boundary=True, volume_perservation=True, override=True)

        # Sample random transformation
        #theta = T.sample_transformation(N)
        theta = T.sample_transformation_with_prior(n_sample=N*len(file_names), mean=None, length_scale=0.1, output_variance=var)
        print(np.shape(theta))

        for f in range(len(file_names)):

                print("cpab ", f, "/", len(file_names))

                file_name = file_names[f]
                mask_name = mask_names[f]

                file_name_noext = os.path.splitext(file_name)[0].split('/')[-1]
                mask_name_noext = os.path.splitext(mask_name)[0].split('/')[-1]
                #print("File name noext: "+file_name_noext)

                data = io.imread(file_name)
                data = img_as_float(data) / 255
                #print("Input image shape:", data.shape)

                data_mask = io.imread(mask_name)
                data_mask = img_as_float(data_mask) / 255
                #print("Input mask shape:", data_mask.shape)		
                if len(data_mask.shape) > 2:
                        data_mask = data_mask[:,:,0]
                data = np.dstack((data, data_mask))
                #print("New image shape:", data.shape)

                data = np.tile(data[None], [N,1,1,1]) # create batch of data

                # Convert data to the backend format
                data = T.backend.to(data, device=args_device)

                # Pytorch have other data format than tensorflow and numpy, color 
                # information is the second dim. We need to correct this before and after
                data = data.permute(0,3,1,2) if args_backend=='pytorch' else data

                # Transform the images
                t_data = T.transform_data(data, theta[N*f:N*(f+1)], outsize=(256, 256)) * 255

                # Get the corresponding numpy arrays in correct format
                t_data = t_data.permute(0,2,3,1) if args_backend=='pytorch' else t_data
                t_data = T.backend.tonumpy(t_data)

                for k in range(N):
                        output = t_data[k]
                        output_ubyte = img_as_ubyte(output[:,:,0:3])
                        io.imsave('../Data/'+save_dir+'/'+subset+'/images/'+file_name_noext+'_'+str(k)+'.png', output_ubyte)
                        output_mask = np.clip(output[:,:,3], -1, 1)
                        output_mask[output_mask > 0.0] = 1.0
                        output_mask_ubyte = img_as_ubyte(output_mask)
                        io.imsave('../Data/'+save_dir+'/'+subset+'/gts/'+mask_name_noext+'_'+str(k)+'.png', output_mask_ubyte, check_contrast=False)

def cpab(file_dir, mask_dir, essais, save_dir, subset, n, var):

        ext = '_%.3f_%.3f' % (n, var)
        save_dir = save_dir+ext
        
        file_names = fetch_file_names(file_dir)
        mask_names = fetch_file_names(mask_dir)

        if not os.path.exists('../Data/%s/%s/images/' % (save_dir, subset)):
                os.makedirs('../Data/%s/%s/images/' % (save_dir, subset))
        if not os.path.exists('../Data/%s/%s/gts/' % (save_dir, subset)):
                os.makedirs('../Data/%s/%s/gts/' % (save_dir, subset))

        cpab_transfo(file_names, mask_names, essais, save_dir, subset, n, var)
