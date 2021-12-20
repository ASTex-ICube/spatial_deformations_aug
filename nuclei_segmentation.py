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

# Code based on:
# F. Mahmood, D. Borders, R. J. Chen, G. N. Mckay, K. J. Salimian,
# A. Baras, N. J. Durr, Deep adversarial training for multi-organ
# nuclei segmentation in histopathology images,
# IEEE Transactions on Medical Imaging
# 39 (11) (2020) 3257–3267. doi:10.1109/TMI.2019.2927182
# Implementation:
# https://github.com/mahmoodlab/NucleiSegmentation
# See the README to get the pre-trained model

import os
import ntpath
import sys
import cv2
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import*

from skimage import io

#python test.py --dataroot <datapath> --name NU_SEG --gpu_ids 0 --display_id 0 
#--loadSize 256 --fineSize 256

def draw_nuclei(image_orig, image_seg, im_size, nuclei_on_image_save_path,
	            nuclei_map_save_path, nuclei_centers_save_path):

        gray = cv2.cvtColor(image_seg, cv2.COLOR_BGR2GRAY)

        # Denoising and thresholding
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 41, 9)
        _, gray = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

        # Only external contours
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        #https://www.programcreek.com/python/example/89409/cv2.fitEllipse
        #https://www.geeksforgeeks.org/python-opencv-cv2-ellipse-method/

        mask = np.zeros((image_orig.shape[0], image_orig.shape[1]), dtype=np.float)

        #image_orig_copy = image_orig.astype(np.uint8).copy()
        image_orig_copy = image_orig.astype(np.int32).copy()

        #image_orig_copy = cv2.resize(image_orig_copy.astype(np.uint8), im_size, interpolation=cv2.INTER_CUBIC)
        #mask = cv2.resize(mask.astype(np.uint8), im_size, interpolation=cv2.INTER_NEAREST)
        sf = 256.0 / image_orig.shape[0]

        centers = []
        for c in cnts:
                if cv2.contourArea(c) > 20 and len(c) > 5:
                        ellipse = cv2.fitEllipse(c)
                        # Gets ellipse center
                        centerE = ellipse[0]
                        # Gets width and height of rotated ellipse (minor, major)
                        widthE = ellipse[1][0]
                        heightE = ellipse[1][1]
                        # Gets rotation angle
                        angleE = ellipse[2]
                        if heightE > 10 and heightE < 70 and (heightE / widthE < 3):
                                center = (int(centerE[0]*sf), int(centerE[1]*sf))
                                centers.append(center)
                                cv2.ellipse(image_orig_copy, ellipse, (0, 255, 0), 4)
                                cv2.ellipse(mask, ellipse, (255, 255, 255), -1)

        image_orig_copy = cv2.resize(image_orig_copy.astype(np.uint8), im_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask.astype(np.uint8), im_size, interpolation=cv2.INTER_NEAREST)

        #io.imsave(nuclei_on_image_save_path, image_orig_copy.astype(np.uint8))
        io.imsave(nuclei_map_save_path, mask.astype(np.uint8))

        centers = np.array(centers).astype(int)
        np.savetxt(nuclei_centers_save_path, centers, fmt='%d')

def my_save_images(output_dir_name, visuals, image_path):
	image_dir = output_dir_name
	short_path = ntpath.basename(image_path[0])
	name = os.path.splitext(short_path)[0]
	
	print("Process image", name)
	
	image_orig = util.tensor2im(visuals['real_A'])
	image_seg = util.tensor2im(visuals['fake_B'])
	
	im_size = (256, 256)
	
	nuclei_seg_name = '%s_nuclei_seg.png' % (name)
	nuclei_seg_save_path = os.path.join(image_dir, nuclei_seg_name)
	#util.save_image(image_seg, nuclei_seg_save_path)
	
	nuclei_on_image_name = '%s_nuclei_overlay.png' % (name)
	nuclei_on_image_save_path = os.path.join(image_dir, nuclei_on_image_name)
	
	nuclei_map_name = '%s_nuclei_map.png' % (name)
	nuclei_map_save_path = os.path.join(image_dir, nuclei_map_name)
	
	nuclei_centers_name = '%s_nuclei_centers.txt' % (name)
	nuclei_centers_save_path = os.path.join(image_dir, nuclei_centers_name)
	
	draw_nuclei(image_orig, image_seg, im_size, nuclei_on_image_save_path,
	            nuclei_map_save_path, nuclei_centers_save_path)

def main(sub_dir, subset, baseline=False):

        if baseline :
                save_dir = '../Data/' + sub_dir + '/nuclei_segmentation_results' + '_baseline'
                image_dir = '../Data/' + sub_dir + '/baseline/%s/images' % subset
        else:
                save_dir = '../Data/' + sub_dir + '/nuclei_segmentation_results'
                image_dir = '../Data/' + sub_dir + '/%s/images' % subset
                
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        
        opt = TestOptions().parse()
        opt.name = 'NU_SEG'
        opt.loadSize = 1024
        opt.fineSize = 1024

        opt.dataroot = image_dir
        opt.results_dir = save_dir
	
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)
	
        # test
        print(dataset)
        for i, data in enumerate(dataset):
                if i >= opt.how_many:
                        break
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                my_save_images(opt.results_dir, visuals, img_path) 

print("let's go")
main('patches_cropped_20', 'train', baseline=False)
