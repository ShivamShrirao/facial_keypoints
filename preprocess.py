import pandas as pd
import numpy as np
import cv2

class FacialDataset:
	def __init__(csv_file):
		self.df=pd.read_csv(csv_file)
		self.image_names=df.iloc[:,0]
		self.key_pts=df.iloc[:,1:].values.reshape(len(self.df),-1,2)
		self.images=[]
		self.rsimg=[]
		self.rskpt=[]

	def load_images(self,image_path):
		for nm in self.image_names:
			img=cv2.imread(image_path+nm)[:,:,:3]
			gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			self.images.append(gray)
		self.images=np.asarray(self.images)

	def resize(self,shape):
		for im,kpt in zip(self.images,self.key_pts):
			x,y=im.shape[:2]
			im=cv2.resize(im,shape)
			rx,ry=shape[0]/x,shape[1]/y
			kpt[:,0]*=ry
			kpt[:,1]*=rx
			self.rsimg.append(im)
			self.rskpt.append(kpt)
		self.rsimg=np.asarray(self.rsimg)
		self.rskpt=np.asarray(self.rskpt)
