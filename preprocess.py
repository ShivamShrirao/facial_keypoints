import pandas as pd
import cv2

class FacialDataset:
	def __init__(csv_file,image_path):
		self.df=pd.read_csv(csv_file)
		self.image_names=df.iloc[:,0]
		self.key_pts=df.iloc[:,1:].values.reshape(len(self.df),-1,2)