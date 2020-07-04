import scipy
import numpy as np
import cv2
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image

from utils import *

# Constants
DEMO_IMAGE_PATH = './test_images/flower_valley.jpg'

# Loop's hyperparameters
STEP = 0.01          # Value of the gradient ascent step
NUM_OCTAVE = 3       # Octave amount
OCTAVE_SCALE = 1.4   # Scale of each octave (look at the picture at the end of notebook)
ITERATIONS = 20      # Iterations amount on the each octave
MAX_LOSS = 10.       # Max loss that prevent ugly artifacts

#Set learning phase = 0 (test mode). Prevent model learning for safety reasons.
K.set_learning_phase(0) 

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# Layers could be from mixed0 up to mixed10
# Check names with command model.summary()
layer_contribution = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5
}

# Loss calculation (L2_norm of layers outputs)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contribution:
    coeff = layer_contribution[layer_name]
    # Get layer output
    activation = layer_dict[layer_name].output
    # Calculate L2 norm
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation)) / scaling  #test without -2:2 and so on

##############GRADINET ASCENDING##############

## Create K.function to fetch loss and gradients from model input

# Create tensor for result storing
dream = model.input 
# Fetch gradient and normalize it
grads = K.gradients(loss = loss, variables = dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  #1e-7 - safety measure to avoid division by 0
# And now we provide link between current result and his gradients with losses
fetch_loss_and_grads = K.function([dream], [loss, grads])

# Gradient ascent's helper function
def eval_loss_and_grads(x):
    '''
    Extract loss and grads values from current input state
    Input:
        Current state of model.input
    Output:
        Current values of loss and input gradients
    '''
    # Pass input x (it will be model.input) through  K.function([model.input], [loss, grads])
    outs = fetch_loss_and_grads([x])
    # Get values
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
    
# Main function
def gradient_ascent(x, iterations, step, max_loss=None, verbose=True):
    '''
    Performs the specified number of gradient ascent steps.
    '''
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if verbose:
            print('Loss value at {} step: {:.3f}'.format(i, loss_value))
        if max_loss is not None and loss_value > max_loss:
            print('Current loss = {:.3f} exceeded max_loss = {:.d}, ascent was finished'.format(loss_value, max_loss))
            break
        x += step * grad_values
    return x
    
    
img = preprocess_img(DEMO_IMAGE_PATH)

# Create list of shapes correspond with octave scales 
original_shape = img.shape[1:3]
octave_shapes = [original_shape]
for i in range(NUM_OCTAVE):
    scaled_shape = tuple([int(dim/(OCTAVE_SCALE ** i)) for dim in original_shape])
    octave_shapes.append(scaled_shape)
octave_shapes = octave_shapes[::-1]


orginal_img = np.copy(img)

# Initialize shrunck image by the smallest image
shrunck_original_img = resize_img(img, octave_shapes[0]) 

for shape in octave_shapes:
    print('Processing image shape: ', shape)
    # Image gradient ascenting 
    img = resize_img(img, shape)
    img = gradient_ascent(img, ITERATIONS, STEP, MAX_LOSS)
    # Lost detail computation
    upscaled_shrunck_original_img = resize_img(shrunck_original_img, shape)
    same_original_size = resize_img(orginal_img, shape)
    lost_detail = same_original_size - upscaled_shrunck_original_img
    # Impute details
    img += lost_detail
    save_img(img, './notebook_images/scale_{}.png'.format(shape))
    
    shrunck_original_img = resize_img(orginal_img, shape)
    
save_img(img, 'result.png') 
print('Process finished, result was saved in the project root folder')
