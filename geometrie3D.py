##########################################################   TP 4: Géométrie Projective 3D - I   ##########################################################
##########################################################      Master TSI - IMOVI, 2021-2022    ##########################################################
##########################################################               Prof. M.M. Nawaf              ####################################################
##########################################################               Student. Assia CHAHIDI             ###############################################
#Objectif: In this TP, the main goal of it is practice all the knowledge obtain from projectif geometry course. 

#Libraries
import pykitti #library for working with the KITTI dataset
import numpy as np #library to manipulate the matrix and tables in python
import matplotlib.pyplot as plt #library to create static, animated and interctive visualizations 
# Specify the dataset to load
basedir = '/home/creation/Desktop/TD/Mise_En_Oevre/KITTI_SAMPLE/RAW'
data = pykitti.raw(basedir , date = '2011_09_26', drive ='0009', frames=range(0, 50, 1))
first_cam2 = data.get_cam2(0)
third_velo = np.array(data.get_velo(0))#lidar points 
K_cam2 = data.calib.K_cam2  #camera image projection matrix (Intrinsic camera matrix)
##Coordinate transformation
third_velo = third_velo[third_velo[:,0] > 5] #supress the x values of the lidar who are smaller than 5, like that we will suppress the values behind the camera
third_velo[:,3] = 1 #change the forth column to 1
new_column = np.zeros((np.shape(K_cam2)[0],1)) #create a new column of zeros 
K_cam2 = np.append(K_cam2, np.zeros((np.shape(K_cam2)[0],1)), axis = 1)# add this new column to the matrix of our camera
K_cam2 = np.vstack([K_cam2, np.array([0,0,0,1])])# Stack [0,0,0,1] array in sequence with the camera matrix vertically to end up with a matrix of shape (4,4)
##Projection matrix
##point_cam0 = camera 2 image projection matrix  @ velodyne camera trasformation matrix @ transpose the velodyne camera
point_cam0 = K_cam2 @ data.calib.T_cam2_velo @ third_velo.T # the projection matrix of the 3D point x in velodyne Lidar coordinates to a point 'point_cam0' in the second camera image
point_cam0 = np.delete(point_cam0, -1, axis=0) # delete the homogene coordinates 
point_cam0[0,:]=(point_cam0[0,:]/point_cam0[2,:]).astype(int)#devide the projection matrix by z by keeping z for a forward use and transform the pixel values from float to integer 
point_cam0[1,:]=(point_cam0[1,:]/point_cam0[2,:]).astype(int)
point_cam0=point_cam0.T 
##Filter the project points finded by reshaping the matrix as the shape of our image (375, 1242)
point_cam0 = point_cam0[point_cam0[:,0] >= 0]
point_cam0 =  point_cam0[point_cam0[:,0] < 1242]
point_cam0= point_cam0[point_cam0[:,1] >= 0]
point_cam0 = point_cam0[point_cam0[:,1] < 375]
##Visualize our lidar points on 2D image by color mapping the inverse depth "z" float values of our projection matrix using "jet" matplotlib color map 
fig = plt.figure(figsize=[20, 5])
plt.imshow(first_cam2)
sc=plt.scatter(point_cam0[:,0],point_cam0[:,1],point_cam0[:,2], c= 1/point_cam0[:,2],cmap='jet')
plt.title('3D cloud point projection')
plt.show()
#Code achieved the goal of this TP in less than 30 lines 