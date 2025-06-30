import cv2
import numpy as np
from matplotlib import pyplot as plt

# grey_levels = 256
# Generate a test image
# test_image = numpy.random.randint(0,grey_levels, size=(11,11))
#test_image = cv2.imread('C:\\Users\\User\\Desktop\\Final project\\RITE Dataset\\AV_groundTruth\\training\\images\\21_training.tif')
#mask_image = cv2.imread('C:\\Users\\User\\Desktop\\Final project\\RITE Dataset\\AV_groundTruth\\training\\fov\\21_training.jpg')
# test_image = cv2.resize(test_image,(512,512))
# mask_image = cv2.resize(mask_image,(512,512))
#print(test_image.shape)
# Define the window size
# windowsize_r = 0
# windowsize_c = 0
# Crop out the window and calculate the histogram
n = 1
x_train = 1
y_train = 1
def flip_patches(img,mask):
    global n
    global x_train,y_train
    for i in range(0,2):
        img = cv2.flip(img, i)
        mask = cv2.flip(mask, i)
        resize_img = cv2.resize(img,(512,512))
        resize_mask = cv2.resize(mask,(512,512))
        x_train[n] = resize_img
        y_train[n] = resize_mask
#        cv2.imwrite('C:\\Users\\User\\Desktop\\raw material\\image patches\\flip_image'+str(n)+'.jpg',img)
        n +=1
        

def rotate_image(img,mask):
    global n
    global x_train,y_train
    for l in range(1,3):
        img = cv2.rotate(img, l*cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        resize_img = cv2.resize(img,(512,512))
        resize_mask = cv2.resize(mask,(512,512))
        x_train[n] = resize_img
        y_train[n] = resize_mask
#        cv2.imwrite('C:\\Users\\User\\Desktop\\raw material\\image patches\\rotate_image'+str(n)+'.jpg',img)
        n +=1
        




def get_patches(preprocessed_images,gnd_truth,fov_mask):
    global n
    global x_train,y_train
    for j in range(0,20):
        for i in range(128,512,128):
            for r in range(30,preprocessed_images[j].shape[0]-40 , i):
                for c in range(30,preprocessed_images[j].shape[1]-40 , i):
                    window1 = preprocessed_images[j][r:r+i,c:c+i]
                    window2 = gnd_truth[j][r:r+i,c:c+i]
                    window3 = fov_mask[j][r:r+i,c:c+i]
                    # hist = numpy.histogram(window,bins=grey_levels)
                    # local_mean = np.mean(window)
                    # return local_mean
                    m = np.mean(window3)
                    if (m>250):
                        resize_img = cv2.resize(window1,(512,512))
                        resize_mask = cv2.resize(window2,(512,512))
                        x_train[n] = resize_img
                        y_train[n] = resize_mask
#                        cv2.imwrite('C:\\Users\\User\\Desktop\\raw material\\image patches\\crop image\\crop_image'+str(n)+'.jpg',window1)
#                        cv2.imwrite('C:\\Users\\User\\Desktop\\raw material\\image patches\\crop image\\crop_gnd'+str(n)+'.jpg',window2)
                        n += 1
                        flip_patches(window1,window2)
#                        rotate_image(window1,window2)
#                         cv2.imshow('crop_mask'+str(r),window2)
#                         cv2.waitKey(0)
#                         cv2.destroyAllWindows()
    return x_train,y_train
                    
def data_augmentation(preprocessed_images,gnd_truth,fov_mask):
    global n
    n = 0
    global x_train,y_train
    x_train = np.zeros((1120,512,512,3),dtype =np. uint8)
    y_train = np.zeros((1120,512,512,3),dtype = np.uint8)
    x_train,y_train = get_patches(preprocessed_images,gnd_truth,fov_mask)
    print(n)
    return x_train,y_train
#    return preprocessed_images_patch,gnd_truth_patch

