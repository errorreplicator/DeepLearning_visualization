from models import manipulation, kerasmodels
from data import dogcat
from tasking import general
from tensorflow.contrib.keras import backend as K
import numpy as np
resolution = 50
epoch = 10
input_size = 1
shape = (resolution,resolution,1)


model = kerasmodels.modelLeNet(shape,1)

for _ in range(5):
    model.pop()

layer_name = 'forVisual'
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
kept_filters = []

layer_output = layer_dict[layer_name].output
# print(layer_output) ### Tensor("forVisual/BiasAdd:0", shape=(?, 25, 25, 50), dtype=float32)
filter_index = 0
loss = K.mean(layer_output[:, :, :, filter_index])
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
# print(K.gradients(loss, input_img))
grads = K.gradients(loss, input_img)[0]
# print(grads)

grads = normalize(grads)

step = 1.

input_img_data = np.random.random((1, resolution, resolution, 1))
print(input_img_data.shape)
input_img_data = (input_img_data - 0.5) * 20 + 128
print(input_img_data.shape)
