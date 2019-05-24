import numpy as np
from tensorflow.contrib.keras import backend as K

def simple_norm(x):
    return x/255

def simple_reshape(x,resolution=50):
    return (np.array(x).reshape(-1, resolution, resolution, 1))

def simple_reshape_color(x,resolution=50):
    return (np.array(x).reshape(-1, resolution, resolution, 3))

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

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None,plot_title=None):
    import matplotlib.pyplot as plt
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        # if (ims.shape[-1] != 3): # uncomment if RGB ???
            # ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        if plot_title is not None:
            plt.suptitle(f'{plot_title}')
        plt.imshow(ims[i], interpolation=None if interp else 'none',cmap='gray')
    plt.show()