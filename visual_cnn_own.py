from models import manipulation, kerasmodels


shape = (50,50,1)

model = kerasmodels.modelLeNet(shape,1)

model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.compile(optimizer='Adam',metrics=['accuracy'],loss='binary_crossentropy')
print(model.summary())
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print(layer_dict)

# for lay in model.layers:
#     print(lay)

# the name of the layer we want to visualize
layer_name = 'forVisual'
# this is the placeholder for the input images
input_img = model.input
# get the symbolic outputs of each "key" layer.
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
kept_filters = []

print(input_img)