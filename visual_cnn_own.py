from models import manipulation, kerasmodels
from time import time
from scipy.misc import imsave
import numpy as np
from tensorflow.contrib.keras import backend as K

resolution = 50
shape = (resolution,resolution,1)

model = kerasmodels.modelLeNet(shape,1)

model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.compile(optimizer='Adam',metrics=['accuracy'],loss='binary_crossentropy')
print(model.summary())
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# print(layer_dict)

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

# the name of the layer we want to visualize
layer_name = 'forVisual'
# this is the placeholder for the input images
input_img = model.input
# get the symbolic outputs of each "key" layer.
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
kept_filters = []


for filter_index in range(0, 50):
    print('start')
    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output

    loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient by its L2 norm
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, resolution, resolution, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    print('end')

    # we will stich the best 36 filters on a 6 x 6 grid.
    n = 6

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 36 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 6 x 6 filters of size 28 x 28, with a 5px margin in between
    margin = 5
    width = n * resolution + (n - 1) * margin
    height = n * resolution + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(resolution + margin) * i: (resolution + margin) * i + resolution,
        (resolution + margin) * j: (resolution + margin) * j + resolution, :] = img

# save the result to disk
imsave('repo/lenet_filters_%dx%d.png' % (n, n), stitched_filters)