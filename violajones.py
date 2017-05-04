'''
XANDER BEBERMAN
CMSC 254 WINTER 2017
HW4 VIOLA-JONES FACE DETECTION
PYTHON 2.7
'''

import numpy as np
import cv2
import time
import csv

# width, height of image
IMG_SIZE = 64

# number of items to take from each bin for training
# e.g. 1000 faces, 1000 backgrounds
TRAIN_SIZE = 2000

t1 = time.time()

### INTEGRAL IMAGES

def make_ii(img):
	'''
	calculates integral image from an array
	'''
	ii = np.empty((IMG_SIZE, IMG_SIZE), int)
	# first row
	ii[0,0] = img[0,0]
	for j in xrange(1,IMG_SIZE):
		ii[0,j] = ii[0,j-1] + img[0,j]
	# rest of image
	for i in xrange(1,IMG_SIZE):
		ii[i,0] = ii[i-1,0] + img[i,0]
		for j in xrange(1,IMG_SIZE):
			ii[i,j] = ii[i-1,j] + ii[i,j-1] - ii[i-1,j-1] + img[i,j]
	return ii

def make_ii_vec(img):
	'''
	makes an integral image for a vector of images (y, x, n)
	'''
	ii = np.empty((IMG_SIZE, IMG_SIZE, TRAIN_SIZE*2), int)
	# first row
	ii[0,0,:] = img[0,0,:]
	for j in xrange(1,IMG_SIZE):
		ii[0,j,:] = ii[0,j-1,:] + img[0,j,:]
	# rest of image
	for i in xrange(1,IMG_SIZE):
		ii[i,0,:] = ii[i-1,0,:] + img[i,0,:]
		for j in xrange(1,IMG_SIZE):
			ii[i,j,:] = ii[i-1,j,:] + ii[i,j-1,:] - ii[i-1,j-1,:] + img[i,j,:]
	return ii

def check_ii(img, ii):
	'''
	check the validity of an integral image
	don't use, very slow
	'''
	for y in xrange(0,IMG_SIZE):
		for x in xrange(0,IMG_SIZE):
			checksum = 0
			for i in xrange(0,y+1):
				for j in xrange(0,x+1):
					checksum += img[i,j]
			if ii[y,x] != checksum:
				print (x,y)
				return 0
	return 1


### FEATURE COMPUTATION

def compute_feature(ftr, iimg):
	'''
	compute one feature for one integral image
	'''
	ftrsum = 0
	for f in ftr:
		(b,a) = f[1]
		(y,x) = f[2]
		if b == 0:
			if a == 0:
				ftrsum += f[0]*iimg[y,x]
			else:
				ftrsum += f[0]*(iimg[y,x] - iimg[y,a-1])
		elif a == 0:
			ftrsum += f[0]*(iimg[y,x] - iimg[b-1,x])
		else:
			ftrsum += f[0]*(iimg[y,x] - iimg[b-1,x] - iimg[y,a-1] + iimg[b-1,a-1])
	return ftrsum

def check_feature(feature, image):
	'''
	checks to see if the computation of a feature from compute_feature is correct
	'''
	iimg = make_ii(image)
	total = 0
	for f in feature:
		p = f[0]
		(b,a) = f[1]
		(y,x) = f[2]
		for j in xrange(b,y+1):
			for i in xrange(a,x+1):
				total += p*image[j,i]
	comp = compute_feature(feature, iimg)
	return comp == total

def compute_feature_vec(ftr, iimg):
	'''
	compute one feature for a vector of integral images
	'''
	ftrsum = np.zeros(iimg.shape[2], int)
	for f in ftr:
		(b,a) = f[1]
		(y,x) = f[2]
		if b == 0:
			if a == 0:
				ftrsum += f[0]*iimg[y,x,:]
			else:
				ftrsum += f[0]*(iimg[y,x,:] - iimg[y,a-1,:])
		elif a == 0:
			ftrsum += f[0]*(iimg[y,x,:] - iimg[b-1,x,:])
		else:
			ftrsum += f[0]*(iimg[y,x,:] - iimg[b-1,x,:] - iimg[y,a-1,:] + iimg[b-1,a-1,:])
	return ftrsum


### FEATURE GENERATION

def make_topbottom(stride = 4, increase = 4):
	'''
	generate list of top/bottom rectangle features
	'''
	features = []
	for w in xrange(8, IMG_SIZE, increase):
		for h in xrange(4, IMG_SIZE, increase):
			for x in xrange(0, IMG_SIZE-w, stride):
				for y in xrange(0, IMG_SIZE-h, stride):
					features.append([(1,(y, x),(y+h/2-1, x+w-1)), (-1,(y+h/2, x),(y+h-1, x+w-1))])
	return features

def make_twoside(stride = 4, increase = 4):
	'''
	generate list of left/right rectangle features
	'''
	features = []
	for w in xrange(4, IMG_SIZE, increase):
		for h in xrange(8, IMG_SIZE, increase):
			for x in xrange(0, IMG_SIZE-w, stride):
				for y in xrange(0, IMG_SIZE-h, stride):
					features.append([(1,(y,x),(y+h-1,x+w/2-1)), (-11,(y,x+w/2),(y+h-1,x+w-1))])
	return features

def make_threeside(stride = 6, increase = 6):
	'''
	generate list of side/middle/side rectangle features
	'''
	features = []
	for w in xrange(12, IMG_SIZE, increase):
		for h in xrange(6, IMG_SIZE, increase):
			for x in xrange(0, IMG_SIZE-w, stride):
				for y in xrange(0, IMG_SIZE-h, stride):
					features.append([(1,(y, x),(y+h-1, x+w/3-1)), (-1,(y, x+w/3),(y+h-1, x+2*w/3-1)), (1,(y, x+2*w/3),(y+h-1, x+w-1))])
	return features

def make_quad(stride = 4, increase = 4):
	'''
	generate list of checkerboard features
	'''
	features = []
	for w in xrange(8, IMG_SIZE, increase):
		for h in xrange(8, IMG_SIZE, increase):
			for x in xrange(0, IMG_SIZE-w, stride):
				for y in xrange(0, IMG_SIZE-h, stride):
					features.append([(1,(y, x),(y+h/2-1, x+w/2-1)), (-1,(y, x+w/2),(y+h/2-1, x+w-1)),
						(-1,(y+h/2, x),(y+h-1, x+w/2-1)), (1,(y+h/2, x+w/2),(y+h-1, x+w-1))])
	return features

def printfeature(feature, polarity, name):
	'''
	exports an image of a feature
	'''
	rect = np.full((IMG_SIZE,IMG_SIZE), 128, np.uint8)
	for f in feature:
		p = 128+127*polarity*f[0]
		(b,a) = f[1]
		(y,x) = f[2]
		cv2.rectangle(rect, (a,b), (x,y), p, -1)
	cv2.imwrite(name, rect)


### CHECKING FOR FACES

def evaluate(ii, fc, fa, fp, ft):
	'''
	given an integral image and a classifier, evaluate whether the image is a face
	'''
	total = 0
	for i in xrange(0,len(fc)):
		total += fa[i]*compute_feature(fc[i], ii)*fp[i]
	return total >= ft

def evaluate_vec(ii, fc, fa, fp, ft):
	'''
	given vector of integral images and a classifier, evaluate the vector for faces
	'''
	total = np.zeros(ii.shape[2])
	for i in xrange(0,len(fc)):
		total += fa[i]*compute_feature_vec(fc[i], ii)*fp[i]
	return total >= ft



### ADABOOST ROUND

def train(nrounds, ftrs, featurelists):
	'''
	given a max number of rounds, list of features, and array of features computed over images, create a classifier
	if nrounds = -1, function will continue until the classifier has < 30 percent FPR
	If the classifier hits this before, the function will stop
	'''
	# get number of positive and negative examples
	(nfeatures, nexamples) = featurelists.shape
	npos = TRAIN_SIZE
	nneg = nexamples - TRAIN_SIZE
	# print to be nice
	print "training with " + str(nfeatures) + " features over " + str(nexamples) + " examples:"
	# create vector of correct classifications
	y = np.ones(nexamples, int)
	y[npos:nexamples] = 0

	# initialize final variables: Features, Indices, Threshold, Polarities, Alphas
	final_f = []
	final_i = []
	final_t = 0
	final_p = []
	final_a = []

	# initialize weights
	weight = np.empty(nexamples)
	weight[0:npos] = 1.0/npos
	weight[npos:nexamples] = 1.0/nneg

	# initialize array for optimal polarities and thetas
	polarity = np.empty(nfeatures, int)
	theta = np.empty(nfeatures, int)

	# initialize arrays for saving errors
	errors = np.empty((nfeatures, nexamples), int)
	errors_weight = np.empty(nfeatures)
	error_temp = np.empty(nexamples, int)

	# for each round
	cont = True
	r = 0
	while cont:
		start_time = time.time()
		# normalize weights
		weight *= 1.0/sum(weight)
		t_pos = sum(weight*y)
		t_neg = 1-t_pos
		# print to be nice!
		print "\tround " + str(r+1) + ":"

		# for each feature
		for f in xrange(0,nfeatures):
			# initialize positive, negative to the left total weight
			s_pos = 0
			s_neg = 0
			# set weighted error to an impossibly high value
			errors_weight[f] = 2
			# get permutation that sorts the computed features
			fsort = np.argsort(featurelists[f,:])
			# for each spot between examples
			for n in xrange(0,nexamples-1):
				# n is the sorted index
				# i is index of the corresponding example
				i = fsort[n]
				# increment weights to the left
				if y[i]:
					s_pos += weight[i]
				else:
					s_neg += weight[i]
				# find weighted errors
				e_pos = s_pos + t_neg - s_neg
				e_neg = s_neg + t_pos - s_pos
				# store best threshold, polarity
				if e_neg < errors_weight[f]:
					errors_weight[f] = e_neg
					polarity[f] = -1
					theta[f] = (featurelists[f,i] + featurelists[f,fsort[n+1]])/2
				if e_pos < errors_weight[f]:
					errors_weight[f] = e_pos
					polarity[f] = 1
					theta[f] = (featurelists[f,i] + featurelists[f,fsort[n+1]])/2
			# compute the actual errors
			errors[f,0:npos] = (polarity[f]*featurelists[f,0:npos]) < polarity[f]*theta[f]
			errors[f,npos:nexamples] = (polarity[f]*featurelists[f,npos:nexamples]) > polarity[f]*theta[f]
			errors_weight[f] = sum(errors[f,:]*weight)

		# get best feature
		best = np.argmin(errors_weight)
		best_error = errors_weight[best]
		# save best feature and associated data to respective lists
		final_i.append(best)
		final_f.append(ftrs[best])
		final_p.append(polarity[best])
		# get beta value
		# this if statement is just a redundancy in case of perfect classifiers
		if best_error == 0:
			beta = 0.1
			# store alpha value
			final_a.append(1)
		else:
			beta = best_error/(1-best_error)
			# store alpha value
			final_a.append(np.log(1.0/beta))
		# update weights
		for n in xrange(0,nexamples):
			weight[n] *= beta**(1-errors[best, n])
		
		# print things to be nice
		end_time = time.time()
		print "\tdone in " + str(end_time-start_time) + " seconds"
		print "\tbest classifier number " + str(best) + " with " + str(sum(errors[best,:])) + " errors and weighed error " + str(best_error)
		print "\tpolarity " + str(polarity[best])
		
		# get evaluation of final classifier so far
		feature_temp = np.zeros(nexamples)
		# compute weighted features for classifier for each image
		for s in xrange(0,r+1):
			i = final_i[s]
			feature_temp += final_a[s]*final_p[s]*featurelists[i,:]
		# get final threshold
		final_t = min(feature_temp[0:npos])
		# find errors
		error_temp[0:npos] = feature_temp[0:npos] < final_t
		error_temp[npos:nexamples] = feature_temp[npos:nexamples] >= final_t
		# get number of false positives
		err = sum(error_temp)
		print "\tfalse positives so far: " + str(err) + " or " + str(float(err)/nneg) + "\n"
		# increment r, continuity variable
		r += 1
		if (float(err)/nneg) < 0.3:
			break
		if nrounds != -1:
			cont = r < nrounds
	# output = the items to keep in the training set
	output = np.logical_not(error_temp)
	return (final_f, final_a, final_p, final_t, final_i, output)



### CLASSIFIER CASCADE

def cascade(nstages, nrounds, ftrs, iimg, export_features = True):
	'''
	given a max number of stages, max number of rounds, vector of features, and array of integral images,
	create a cascade of classifiers
	if nstages = -1, function will continue until cascade hits <1 percent FPR
	If the cascade hits this before, the function will stop.
	'''
	# initialize final variables
	features = []
	alphas = []
	polarities = []
	thresholds = []

	# stage counter and continuity variable
	n = 0
	cont = True
	# for each stage
	while cont:
		print "stage " + str(n) + ":"
		# calculate results of feautres for all training images
		nfeatures = len(ftrs)
		nexamples = iimg.shape[2]
		featurelists = np.empty((nfeatures, nexamples), int)
		for i in xrange(0,nfeatures):
			featurelists[i,:] = compute_feature_vec(ftrs[i], iimg)
		
		# do adaboost/one stage
		(final_f, final_a, final_p, final_t, final_i, output) = train(nrounds, ftrs, featurelists)
		
		# append classifier info to final variables
		features.append(final_f)
		alphas.append(final_a)
		polarities.append(final_p)
		thresholds.append(final_t)
		
		# remove easily classified examples
		output[TRAIN_SIZE:] = output[TRAIN_SIZE:] == 0
		featurelists = featurelists[:,output[:] == 1]
		iimg = iimg[:,:,output[:] == 1]
		
		# export features if desired
		if export_features:
			nf = len(final_f)
			for i in xrange(0,nf):
				printfeature(final_f[i], final_p[i], "features/ftr"+str(n)+"-"+str(i)+".jpg")
		# remove used features from the list of features to avoid cycles
		ftrs = np.delete(ftrs, final_i)
		print ""
		# increment stage, continuity variables
		if float(nexamples-TRAIN_SIZE)/TRAIN_SIZE < 0.01:
			break
		n += 1
		if nstages != -1:
			cont = n < nstages
	return (features, alphas, polarities, thresholds)


def evaluate_cascade(ii, features, alphas, polarities, thresholds):
	'''
	given a cascade, evaluate whether an integral image is a face
	'''
	n = len(features)
	for i in xrange(0,n):
		x = evaluate(ii, features[i], alphas[i], polarities[i], thresholds[i])
		if (not x):
			return 0
	return 1


### FACE DETECTION

def detect_overlap(x, y, faces):
	'''
	given x,y value and list of faces, detect if there is substantial overlap
	'''
	m = IMG_SIZE/4
	for f in faces:
		if (y < f[0] + IMG_SIZE-m) and (x < f[1] + IMG_SIZE-m) and (x > f[1]):
			return f[1] + IMG_SIZE - x - m
		if (y < f[0] + IMG_SIZE) and (f[1] < x + IMG_SIZE-m) and (f[1] > x):
			return x + IMG_SIZE - f[1] - m
	return 0

def detect(img, fc, fa, fp, ft, stride = 8, name = "faces.jpg"):
	'''
	detect faces in an image and output an image of the faces in rectangles
	'''
	print "detecting faces..."
	out = img
	(h,w) = img.shape
	faces = []
	for y in xrange(0, h-IMG_SIZE, stride):
		x = 0
		while x < w-IMG_SIZE:
			iimg = make_ii(img[y:y+IMG_SIZE, x:x+IMG_SIZE])
			if evaluate_cascade(iimg, fc, fa, fp, ft):
				delta = detect_overlap(x, y, faces)
				if delta != 0:
					x += delta
				else:
					print "\tfound a face at " + str(x) + ", " + str(y)
					faces.append((y,x))
					cv2.rectangle(out, (x,y), (x+IMG_SIZE,y+IMG_SIZE), 255, 1)
					x += IMG_SIZE
			else:
				x += stride
	cv2.imwrite(name, out)
	print "\n"
	return faces


### OUR MAIN FUNCTION
if __name__ == '__main__':
	# importing faces
	print "importing images..."
	images = np.empty((IMG_SIZE, IMG_SIZE, TRAIN_SIZE*2), int)
	for i in xrange(0,TRAIN_SIZE):
		images[:, :, i] = cv2.imread("faces/face" + str(i) + ".jpg", 0)
	for i in xrange(0, TRAIN_SIZE):
		images[:, :, TRAIN_SIZE+i] = cv2.imread("background/" + str(i) + ".jpg", 0)

	# make integral image
	print "creating integral images..."
	iimages = make_ii_vec(images)

	# make all the features
	print "computing features..."
	ftrs = make_topbottom() + make_threeside() + make_quad() + make_twoside()

	# get number of features
	ftrlen = len(ftrs)

	# get classifier cascade
	(fc, fa, fp, ft) = cascade(10, 250, ftrs, iimages)

	# import testing image
	bigimg = cv2.imread("class.jpg", 0)

	# find faces in image
	detect(bigimg, fc, fa, fp, ft)

	t2 = time.time()
	print t2-t1
