import numpy as np
import scipy
import cv2
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K

def preprocess_img(img_path):
    '''
    Convert raw image to the InceptionV3 input format
    Input:
        Path to file, str
    Output:
        Numpy arrray with shape [1, img_height, img_width, 3]
    '''
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def resize_img(img, size):
    '''
    Resize input image by zooming it
    Input:
        Numpy matrix with shape [1, img_height, img_width, 3]
    Output:
        Numpy matrix with shape [1, zoomed_img_height, zoomed_img_width, 3]
    '''
    img = np.copy(img)
    factor = (1, float(size[0])/img.shape[1], float(size[1])/img.shape[2], 1)
    return scipy.ndimage.zoom(img, factor, order=1) #order - spline interpolation order. Could be in range(0, 5)

def deprocess_img(x):
    '''
    Convert InceptionV3's result input after gradient ascent to the np.array image
    Input:
        Numpy arrray with shape [1, img_height, img_width, 3]
    Output:
        Numpy arrray with shape [img_height, img_width, 3]
    '''
    if K.image_data_format() == 'channel_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape(x.shape[1], x.shape[2], 3)  
    # Undo InceptionV3 preprocess input
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    

def save_img(img, file_name):
    '''
    Save image into png format
    Input:
        Numpy arrray with shape [1, img_height, img_width, 3]
    Output:
        None
    '''
    pil_img = deprocess_img(np.copy(img))
    cv2.imwrite(file_name, pil_img)
