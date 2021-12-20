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
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

L = [
[76.8923,	75.3258,	76.9841,	76.2099,	78.9876],
[81.8753,	79.2827,	80.6964,	83.7666,	83.2478],
[86.2557,	87.8753,	86.9777,	84.1029,	84.4614],
[83.5204,	86.4659,    85.4879,	86.2816,	83.7561],
[77.1514,	82.3749,	79.9538,	79.3910,	77.6573],
[86.2831,	84.4938,	87.2794,	83.8855,	78.4286],
[78.8660,	86.5380,	79.5826,	79.1694,	81.5264],
[73.3802,	77.8097,    73.9449,	65.0761,	78.8160],
[80.8753,	80.5216,	82.2541,	76.6975,	85.2905],
[84.4554,	86.0077,	86.2467,	82.9848,	82.6894],
[73.9246,	75.8111,	79.3964,	75.8054,	70.3772],
[71.1374,	71.5147,	71.2392,	66.7548,	69.9095]
]

results = []

for i in range(len(L)):
	l = []
	for j in range(len(L)):
		U1, p = mannwhitneyu(L[i], L[j], method='exact')
		#print(U1, U2, p)
		l.append(p < 0.05)
	results.append(l)



for l in results:
	print(l)