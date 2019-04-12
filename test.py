from models import manipulation, kerasmodels
from data import dogcat
from tasking import general
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import models
from scipy.misc import imsave
import numpy as np
import time
#leNet loss: 0.1896 - acc: 0.9255
#mine loss: 0.2064 - acc: 0.9202
resolution = 50
epoch = 10
input_size = 1
shape = (resolution,resolution,1)

# model = kerasmodels.modelSeq1(shape,1)
# X_train, y_train = dogcat.load_data(test_data=False)
# X_train = general.simple_reshape(X_train,50)
# X_train = general.simple_norm(X_train)
# model.fit(X_train,y_train,epochs=10,batch_size=50)
# model.save(filepath='repo/modelSequential.h5')

# model = models.Sequential()
model = models.load_model('repo/modelLeNet.h5')

# for _ in range(5):
#     model.pop()
#
layer_name = 'forVisual'
input_img = model.input
print('input img',input_img)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print(layer_dict)
kept_filters = []
#
# layer_output = layer_dict[layer_name].output
# print(layer_output) ### Tensor("forVisual/BiasAdd:0", shape=(?, 25, 25, 50), dtype=float32)
filter_index = 0
# loss = K.mean(layer_output[:, :, :, filter_index])
# print(loss) ###Tensor("Mean:0", shape=(), dtype=float32)
# print(layer_output[:, :, :, filter_index]) ###Tensor("strided_slice_1:0", shape=(?, 25, 25), dtype=float32)

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
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

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
print(input_img)
# we run gradient ascent for 20 steps
for filter_index in range(0, 50):
    # we scan through the first 50 filters
    print('Processing filter %d' % filter_index)
    start_time = time.time()

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
        print('YESYESYES')
    end_time = time.time()
    # print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
# print('ile filtrow przechowujemy:',len(kept_filters))
# print('print the kept_filters',kept_filters)
n=2
margin = 5
width = n * resolution + (n - 1) * margin
height = n * resolution + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))
print(100*'#')
for i in range(n):
    for j in range(n):
        if (i * n + j) > 1: break
        img, loss = kept_filters[i * n + j]
        stitched_filters[(resolution + margin) * i: (100 + margin) * i + resolution,
        (resolution + margin) * j: (100 + margin) * j + resolution, :] = img
        print(len(kept_filters))
        print(i * n + j)


# save the result to disk
print('saving result')
imsave('repo/lenet_filters_%dx%d_.png' % (n, n), stitched_filters)
print('results saved')