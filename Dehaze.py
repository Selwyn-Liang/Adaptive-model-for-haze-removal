import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
sys.path.append('/home/selwyn/caffe-master/python')
sys.path.append('/home/selwyn/caffe-master/python/caffe')
import caffe
import numpy as np
import math
import time
import cv2
from skimage import transform



def TransmissionEstimate(im_path, height, width):
	caffe.set_mode_cpu()
	net = caffe.Net('deploy/Dehaze_SUN_deploy.prototxt', 'model/gan_iter_95000.caffemodel', caffe.TEST)
        net.blobs['data'].reshape(1,3,height,width) # (batch_size,c,h,w)
        
	transformers = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformers.set_transpose('data', (2,0,1))
	transformers.set_channel_swap('data', (2,1,0))
        #transformers.set_raw_scale('data', 255)
        im = caffe.io.load_image(im_path) 
        transformed_image = transformers.preprocess('data', im)
        net.blobs['data'].data[...] = transformed_image
	out = net.forward()
        images = np.array(out['eltwise_g'])
        channel_swap = (0, 2, 3, 1)
        images = images.transpose(channel_swap)
        #images *= 255.0


	return images[0]


def Guidedfilter(im,p,r,eps):
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	q = mean_a*im + mean_b
	return q


def TransmissionRefine(im,et):
        #gray = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGB2GRAY)
	#gray = np.float64(gray)/255
        #merged = cv2.merge([gray,gray,gray])
	r = 60
	eps = 0.0001
	t = Guidedfilter(im,et,r,eps)
	return t
	
def Recover(im,t):
	res = np.empty(im.shape,im.dtype)
	for ind in range(0,3):
		res[:,:,ind] = im[:,:,ind]-t[:,:,ind]
	return res



if __name__ == '__main__':
	if not len(sys.argv) == 2:
		print 'Usage: python DeHazeNet.py haze_img_path'
		exit()
	else:
		im_path = sys.argv[1]
	src = cv2.imread(im_path)
        height = src.shape[0]
	width = src.shape[1]
        height = height//2//2*2*2
        width = width//2//2*2*2
        #print height
        #print width
        #time.sleep(3)
        if(width!= src.shape[1] or height != src.shape[0]):
             src = transform.resize(src, (height,width))
        start = time.clock()
	te = TransmissionEstimate(im_path, height, width)
        end = time.clock()
        print "read:%f s" % (end - start)
        time.sleep(20)
        #t = TransmissionRefine(src,te)
        #I = src/255.0
        #J = Recover(I,te)
	
	cv2.imshow('TransmissionEstimate',te)
	#cv2.imshow('TransmissionRefine',t)
        #cv2.imshow('Result',J)
	cv2.imshow('Origin',src)
	cv2.waitKey(0)
	save_path = im_path[:-4]+'_Dehaze'+im_path[-4:len(im_path)]
	cv2.imwrite(save_path,te*255)
