import numpy as np
from tensorflow.contrib.keras import backend as K

def simple_norm(x):
    return x/255

def simple_reshape(x,resolution=50):
    return (np.array(x).reshape(-1, resolution, resolution, 1))

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# utility function to normalize a tensor by its L2 norm
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)