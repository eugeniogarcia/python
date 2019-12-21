# Images Processing

```py
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

Dimensions

```sh
(None,28,28,1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
(None,26,26,32)
model.add(layers.MaxPooling2D((2, 2)))
(None,13,13,32)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
(None,11,11,64)
model.add(layers.MaxPooling2D((2, 2)))
(None,5,5,64)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
(None,3,3,64)

model.add(layers.Flatten())
(None,3 *3 * 64)
model.add(layers.Dense(64, activation='relu'))
(None,64)
model.add(layers.Dense(10, activation='softmax'))
(None,10)
```

Parameters

```sh
3*3*32 + 32
Max Pooling does not have weight to be trained
32*3*3*64 + 64
Max Pooling does not have weight to be trained
64*3*3*64 + 64

Flatten does not have weight to be trained
3*3*64*64 + 64
64*10 +10 
```

## Padding & Strides

In Conv2D layers, padding is configurable via the padding argument, which takes two values: "valid", which means no padding (only valid window locations will be used); and "same", which means “pad in such a way as to have an output with the same width and height as the input.” The padding argument defaults to "valid".

The other factor that can influence output size is the notion of strides. The description of convolution so far has assumed that the center tiles of the convolution windows are all contiguous. But the distance between two successive windows is a parameter of the convolution, called its stride, which defaults to 1.

## Dropout

```py
model.add(layers.Dropout(0.5))
```

## Using pre-trained layers

### Importing a model

We add to the layers of our mode a pre-trained model, `VGG16`. We want to leverage the _"convolutional"_ part of the model, so we specify `include_top=False`. The idea behind this technique is that the features to be extrated from the images are the same, and that the element that differs is how the images, or rather the features contained in the images, are classified - the _top layer_.

```py
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) 

model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

```

we can see the model

```py
>>> conv_base.summary()
Layer (type) Output Shape Param #
================================================================
input_1 (InputLayer) (None, 150, 150, 3) 0
________________________________________________________________
block1_conv1 (Convolution2D) (None, 150, 150, 64) 1792
________________________________________________________________
block1_conv2 (Convolution2D) (None, 150, 150, 64) 36928
________________________________________________________________
block1_pool (MaxPooling2D) (None, 75, 75, 64) 0
________________________________________________________________
block2_conv1 (Convolution2D) (None, 75, 75, 128) 73856
________________________________________________________________
block2_conv2 (Convolution2D) (None, 75, 75, 128) 147584
________________________________________________________________
block2_pool (MaxPooling2D) (None, 37, 37, 128) 0
________________________________________________________________
block3_conv1 (Convolution2D) (None, 37, 37, 256) 295168
________________________________________________________________
block3_conv2 (Convolution2D) (None, 37, 37, 256) 590080
________________________________________________________________
block3_conv3 (Convolution2D) (None, 37, 37, 256) 590080
________________________________________________________________
block3_pool (MaxPooling2D) (None, 18, 18, 256) 0
________________________________________________________________
block4_conv1 (Convolution2D) (None, 18, 18, 512) 1180160
________________________________________________________________
block4_conv2 (Convolution2D) (None, 18, 18, 512) 2359808
________________________________________________________________
block4_conv3 (Convolution2D) (None, 18, 18, 512) 2359808
________________________________________________________________
block4_pool (MaxPooling2D) (None, 9, 9, 512) 0
________________________________________________________________
block5_conv1 (Convolution2D) (None, 9, 9, 512) 2359808
________________________________________________________________
block5_conv2 (Convolution2D) (None, 9, 9, 512) 2359808
________________________________________________________________
block5_conv3 (Convolution2D) (None, 9, 9, 512) 2359808
________________________________________________________________
block5_pool (MaxPooling2D) (None, 4, 4, 512) 0
================================================================
Total params: 14714688
```

We can also train the model, or rather, some of the laters of the model. In this example __after__ the layer named `block5_conv1 we make all the layers trainable:

```py
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		set_trainable = True
	if set_trainable:
		layer.trainable = True
	else:
		layer.trainable = False
```

# Text Processing

We are to describe here three techniques to process text:

- One-hot encoding
- Embeding
- Embeding and Recurrent Networks
- Convolution (1D)

## One-hot encode

- Take a sequence of `maxlen` words
- Each word ins one-hot encoded in a 10000 space
- The sequence of maxlen words is converted on an 10000 one-hot encoded vector
	- If the sequence had for example one word two times, that signal is losts - the encodig is still a `1`
	- The order in which the words appear in the sequence is lost


```py
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Dimensions

```sh
(None,10000)
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
(None,16)
model.add(layers.Dense(16, activation='relu'))
(None,16)
model.add(layers.Dense(1, activation='sigmoid'))
(None,1) 
```

Parameters

```sh
10000 * 16 + 16
16*16 + 16
16*1 + 1
```

## Embedding

- Take a sequence of `maxlen` words
- We feed the sequence as it is

```py
model = Sequential()

model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
``` 

Dimensions

```sh
(None,maxlen)
model.add(Embedding(10000, 8, input_length=maxlen))
(None,maxlen,8) 
model.add(Flatten())
(None,maxlen*8)
model.add(Dense(32, activation='relu'))
(None,32)
model.add(Dense(1, activation='sigmoid'))
(None,1) 
```

Parameters

```sh
10000 * 8
Flatten does not have weights to train
(maxlen*8)*32 + 32
32*1 + 1
```

## Recurrent

- Take a sequence of `maxlen` words
- We feed the sequence as it is

Internally each item of the sequence will be feed at a time - hence the name recurrent, and a state - memory - is kept and carried forward with each item of the sequence. Between sequences the state is reset.

So if we have two sequences:

1. [a, b, e, f]
2. [g, b, h, k] 

When we process the firs one, `[a, b, e, f]`, the items are feeded one by one, a, then b, then e and then f, and a state is carried in the training. When we move to use the second sequence, `[g, b, h, k]`, the state is reset.

```py
model = Sequential()

model.add(Embedding(10000, 8))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
``` 

Dimensions

```sh
(None,maxlen)
model.add(Embedding(10000, 8))
(None,maxlen,8)
model.add(LSTM(32))
(None,32)
model.add(Dense(1, activation='sigmoid'))
(None,1) 
```

Parameters

```sh
10000 * 8
8 * 32 + 32 *32 + 32
32*1 + 1
```

### Multiple Recurrent Layers

We can combine several recurrent layers - as we may with any other type of layer, this is not the reason why i´m adding this comment here -. When dealing with recurrent layers we have the option of including in the output the intermidiate outputs or not - that is,in our example we can add to the output of the layer the outputs when the network was feeded with a, b, and e. If we include the intermediate outputs, the recurrent layer could beed another recurrent layer. For example:

```py
model = Sequential()

model.add(Embedding(10000, 8))
model.add(LSTM(32),return_sequences=True)
model.add(LSTM(32),return_sequences=True)
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
``` 

Dimensions

```sh
(None,maxlen)
model.add(Embedding(10000, 8))
(None,maxlen,8)
model.add(LSTM(32),return_sequences=True)
(None,maxlen,32)
model.add(LSTM(32),return_sequences=True)
(None,maxlen,32)
model.add(LSTM(32))
(None,32)
model.add(Dense(1, activation='sigmoid'))
(None,1) 
```

Parameters

```sh
10000 * 8
8 * 32 + 32 *32 + 32
32 * 32 + 32 *32 + 32
32 * 32 + 32 *32 + 32
32*1 + 1
```

### Dropout in recurrent layers

```sh
model = Sequential()

model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None, 10)))
model.add(layers.Dense(1))
```

### Bidirectional Recurrent Layers

```py
model = Sequential()

model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))
```

## Convolution

```py
model = Sequential()

model.add(layers.Embedding(max_features, 128, input_length=200))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
``` 

Dimensions

```sh
(None,200)
model.add(layers.Embedding(max_features, 128, input_length=200))
(None,200,128)
model.add(layers.Conv1D(32, 7, activation='relu'))
(None,194,32)
model.add(layers.MaxPooling1D(5))
(None,38,32)
model.add(layers.Conv1D(32, 7, activation='relu'))
(None,32,32)
model.add(layers.GlobalMaxPooling1D())
(None,32)
model.add(layers.Dense(1))
(None,(1)
```

Parameters

```sh
max_features*128
128*7*32 + 32
Gobal does not have weights to train
32*7*32 +32
Global does not have weights to train
32*1 +1
```