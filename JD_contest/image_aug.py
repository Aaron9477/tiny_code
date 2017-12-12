#importing some useful packages
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
#%matplotlib inline
import matplotlib.image as mpimg
from skimage import data, exposure, img_as_float

def augment_brightness_camera_images(image,brightness):
    #image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    #random_bright = brightness+0.01*np.random.uniform(-1,1)
    random_bright = brightness+0.2*np.random.uniform(-1,1)
    #print(random_bright)
    image1= exposure.adjust_gamma(image, random_bright)
    #image1[:,:,2] = image1[:,:,2]*random_bright
    #image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    if ang_range !=0:
        ang_rot = np.random.uniform(ang_range)-ang_range/2
        rows,cols,ch = img.shape    
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])

        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2

        # Brightness


        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

        shear_M = cv2.getAffineTransform(pts1,pts2)

        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness != 0:
      img = augment_brightness_camera_images(img,brightness)

    return img
def main(old_path,new_path):
    brightrange=[0.6,0.7,1.3,1.7,2.2]
    image = cv2.imread(old_path)
    #img = transform_image(image,20,10,5,brightness=1)
    for brightness in brightrange:# 0.4 0.6 0.8 1.2
        img = transform_image(image,0,0,0,brightness=brightness)
        ext='_'+str(brightness)+'.jpg'
        final_path=new_path.replace('.jpg',ext)
        print(final_path)
        cv2.imwrite(final_path, img)

if __name__ == '__main__':
    rootpath='/home/zq610/WYZ/JD_contest/wipe_out/one_left/good/'
    new_root='/home/zq610/WYZ/JD_contest/wipe_out/good_aug/'

    for (root, dirs, files) in os.walk(rootpath,topdown=True):
        if not os.path.exists(new_root):
            os.mkdir(new_root)
        for d in dirs:
            d = os.path.join(new_root, d)
            if not os.path.exists(d):
                os.mkdir(d)
        for f in files:
            
            old_path = os.path.join(root, f)
            new_path = os.path.join(new_root, old_path.split('/')[-2],f)
            main(old_path,new_path)



