import numpy as np
import scipy
import cv2
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K

##### Image subrutine ####

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


##### Gradient ascending subrutine ####
    
def get_loss(layer_contribution, model):
	'''
	Calculate loss based on the L2_norm of selected layers outputs.
	Input:
		layer_contribution - dict with layers names and their impact weigts.
		Example: {
			'mixed1': 0.3,
			'mixed3': 0.01}
		Names should be the same with the corresponded model layers.
		model - by default it is InceptionV3 from keras.applications.
	Output:
		Calculated L2 loss, tf.Variable.	
	'''
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	loss = K.variable(0.)
	for layer_name in layer_contribution:
		coeff = layer_contribution[layer_name]
		# Get layer output
		activation = layer_dict[layer_name].output
		# Calculate L2 norm
		scaling = K.prod(K.cast(K.shape(activation), 'float32'))
		loss += coeff * K.sum(K.square(activation)) / scaling
	return loss
	
def eval_loss_and_grads(x, fetch_func):
    '''
    Extract loss and grads values from current input state
    Input:
        x - current state of model.input
        fetch_func - K.functions([model.input], [loss, grads])
    Output:
        Current values of loss and input gradients
    '''
    # Pass input x (it will be model.input) through  K.function([model.input], [loss, grads])
    outs = fetch_loss_and_grads([x])
    # Get values
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
