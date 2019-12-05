We are to describe here three techniques to process text:

- One-hot encoding
- Embeding
- Embeding and Recurrent Networks

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
8 * 32 + 8 *32 + 32
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
8 * 32 + 8 *32 + 32
8 * 32 + 8 *32 + 32
8 * 32 + 8 *32 + 32
32*1 + 1
```