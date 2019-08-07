`tf.convert_to_tensor`  

Converts the given value to a Tensor. This function converts Python objects of various types to Tensor objects. It accepts Tensor objects, numpy arrays, Python lists, and Python scalars.  

`tf.strided_slice`  

Extracts a strided slice of a tensor (generalized python array indexing).  

`tf.reshape`  

Reshapes a tensor.  

`tf.transpose`  

Permutes the dimensions according to perm.  

The returned tensor's dimension i will correspond to the input dimension perm[i]. If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor. Hence by default, this operation performs a regular matrix transpose on 2-D input Tensors. If conjugate is True and a.dtype is either complex64 or complex128 then the values of a are conjugated and transposed.  

`tf.cast`  

Casts a tensor to a new type.  

`tf.image.resize_with_crop_or_pad`  

Crops and/or pads an image to a target width and height. Resizes an image to a target width and height by either centrally cropping the image or padding it evenly with zeros. If width or height is greater than the specified target_width or target_height respectively, this op centrally crops along that dimension. If width or height is smaller than the specified target_width or target_height respectively, this op centrally pads with 0 along that dimension.

image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].  

`tf.image.per_image_standardization`  

Linearly scales image to have zero mean and unit variance.

`tf.one_hot(z, 10)`  

Returns a one-hot tensor. 

```
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```

The locations represented by indices in indices take value on_value, while all other locations take value off_value. on_value and off_value must have matching data types. If dtype is also provided, they must be the same data type as specified by dtype. If on_value is not provided, it will default to the value 1 with type dtype. If off_value is not provided, it will default to the value 0 with type dtype. If the input indices is rank N, the output will have rank N+1. The new axis is created at dimension axis (default: the new axis is appended at the end). If indices is a scalar the output shape will be a vector of length depth

