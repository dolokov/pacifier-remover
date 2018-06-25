from __future__ import print_function
import os 
from random import shuffle
from glob import glob 

import numpy as np 
import cv2 

import Settings

class Database(object):
	def __init__(self,dataset_id,mode,shape):
		self.dataset_id = dataset_id
		self.mode = mode 
		self.shape = shape

		self.prepare_images()

	def get_directory_by_dataset_id(self,dataset_id):
		directories = {'pacifier':Settings.directory_pacifier,'wopacifier':Settings.directory_wo_pacifier }
		return directories[dataset_id]

	def prepare_images(self):
		d = self.get_directory_by_dataset_id(self.dataset_id)
		fns = []
		for fe in Settings.file_extensions:
			fns+=glob(os.path.join(d,'*%s'%fe))
		self.data = {}
		for fn in fns:
			im = cv2.imread(fn)
			if im is not None:
				# make square by center cropping
				s = im.shape[:2]
				_s = min(s)
				im = cv2.resize(im,(self.shape[1],self.shape[2]),interpolation=cv2.INTER_AREA)
				self.data[fn] = {'im':im}
				# hflipping
				self.data[fn+'_hflip'] = {'im':np.fliplr(im)}

		self.data_keys = self.data.keys()
		shuffle(self.data_keys)
		self.data_idx = 0 
		self.steps_per_epoch = len(self.data_keys) / self.shape[0]

	def get_next_batch(self,batch_size=None):
		if batch_size is None:
			batch_size = self.shape[0]

		batch = []
		while len(batch) < batch_size:
			batch.append(self.data[self.data_keys[self.data_idx]]['im'])
			self.data_idx = (self.data_idx+1)%len(self.data_keys)
		batch = np.array(batch,np.float32)
		
		batch = (batch/128.) -1. # [0,255]=>[-1,1]
		batch = (batch-batch.mean()) / batch.std() # zero mean, unit std
		#print('batch',batch.shape,'mean',batch.mean(),'std',batch.std(),'minmax',batch.min(),batch.max())

		return batch 

if __name__ == '__main__':
	batch_shape = [8,64,64,3]
	db = Database('pacifier','train',batch_shape)
	b = db.get_next_batch()
	print(b)