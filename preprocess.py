import pandas as pd
import numpy as np
import cv2

class FacialDataset:
	def __init__(self,csv_file):
		self.df=pd.read_csv(csv_file)
		self.image_names=self.df.iloc[:,0]
		self.key_pts=self.df.iloc[:,1:].values.reshape(len(self.df),-1,2).astype(np.float32)
		self.images=[]
		self.rsimg=[]
		self.rskpt=[]

	def load_images(self,image_path):
		for nm,kpt in zip(self.image_names,self.key_pts):
			img=cv2.imread(image_path+nm)[:,:,:3].astype(np.float32)
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			self.images.append(gray.astype(np.float32))

	def resize(self,shape,chnl=1):
		for im,kpt in zip(self.images,self.key_pts):
			x,y=im.shape[:2]
			im=cv2.resize(im,shape)
			rx,ry=shape[0]/x,shape[1]/y
			kpt[:,0]*=ry
			kpt[:,1]*=rx
			self.rsimg.append(im.reshape(*shape,chnl))
			self.rskpt.append(kpt)
		self.rsimg=np.asarray(self.rsimg).astype(np.float32)
		self.rskpt=np.asarray(self.rskpt).astype(np.float32)

	def normalize(self):
		self.rsimg/=255
		self.nkpt=np.empty_like(self.rskpt)
		self.nkpt[:,:,0]/=self.rsimg.shape[1]
		self.nkpt[:,:,1]/=self.rsimg.shape[2]
