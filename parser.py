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

import argparse

def parse_args():
    
	parser = argparse.ArgumentParser(description="UNet Segmentation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--img_height', '-ih', default=256, type=int,
			help="Height of the images")
	parser.add_argument('--img_width', '-iw', default=256, type=int,
			help="Width of the images")
	parser.add_argument('--dataset_name', '-dn', default='patches_cropped_20', type=str,
			help="Which dataset to use")
	
	parser.add_argument('--unet_type', '-ut', default='unet', type=str,
			choices=['unet', 'R2unet', 'attunet', 'inceptionunet', 
			         'unetplusplus', 'segnet', 'segnetpretrained',
					 'uresnet50pretrained'],
			help="Which UNet to use")


	parser.add_argument('--learning_rate', '-lr', default=0.000004, type=float, # 0.000002
			help="Training hyperparameter : learning rate")
	parser.add_argument('--beta', '-b', default=0.5, type=float,
			help="Training hyperparameter : beta")
	parser.add_argument('--batch_size', '-bs', default=8, type=int,
			help="Training hyperparameter : batch size")
	parser.add_argument('--epochs', '-e', default=10, type=int,
			help="Training hyperparameter : number of epochs")

	parser.add_argument('--resume_training', '-rt', default=False, type=bool,
            help="Start from the latest checkpoint available ?")
	parser.add_argument('--test', '-t', default=False, type=bool,
            help="Test the model on the test set ?")
	parser.add_argument('--verbose', '-v', default=False, type=bool,
            help="Print infos during training ?")

	parser.add_argument('--n_grid', '-n', default=3, type=int,
			help="Grid size for gbd")
	parser.add_argument('--sigma', '-s', default=20, type=int,
			help="Sigma for gbd")

	return parser.parse_args()
