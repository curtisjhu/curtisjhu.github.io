

# Getting Dimensions Right

## MLP (Dense) Layers
This is just a list of matrix operations.

$$z_{i+1} = W_i z_i + b_i$$

$(n_{i+1} \times 1) = (n_{i+1} \times n_i) (n_i \times 1) + (n_{i+1} \times 1)$
$(n_{i+1} \times m) = (n_{i+1} \times n_i) (n_i \times m) + (n_{i+1} \times 1)$

This means that $W_i$ has a dimension that transforms $z_i$ dimension into $z_{i+1}$ dimension.


```
model = nn.Sequential(OrderedDict([
    ('dense1', nn.Linear(764, 100)),
    ('act1', nn.ReLU()),
    ('dense2', nn.Linear(100, 50)),
    ('act2', nn.ReLU()),
    ('output', nn.Linear(50, 10)),
    ('outact', nn.Sigmoid()),
])
```

for this model you can pass in inputs of size (m, 764) and first layer turns out size (m, 100)



https://youtu.be/yslMo3hSbqE

## Convolutional Neural Networks
Filters and networks are a bit more tricky

Inputs and outputs are in the shape (batch size, num of channels, height, width)
Note that this is the dimensions for a batch of images...

When you apply a convolution, you are just changing the dimensions of this image. But it still remains an image. Just one we don't interpret very well.

Parameters for defining a convolutional neural network are:
- in channels
- out channels
- kernel size
- stride
- padding

```py
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode, dilation, groups, bias)
```

Obviously, batch size stays the same.
Num of channels should also stay the same.

But to calculate H, W

Pseudo:
Usually dilation is 1.
$$ output = \left\lfloor \frac{input - kernel size + 2 \times padding}{stride} + 1 \right\rfloor$$


$$ H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel size[0] -1) - 1}{stride[0]} + 1 \right\rfloor$$

$$ W_{out} = \left\lfloor \frac{W_{in} + 2 \times padding[1] - dilation[1] \times (kernel size[1] -1) - 1}{stride[1]} + 1 \right\rfloor$$

Intuitively...
IDK

Some animations of these parameters.
https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

```py
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
```

## Max Pooling 
Again the input is an image in the form (batch size, channels, width, height)


Obviously, batches and channels stay the same.

The change in width and height follows the previous example. It's similarly a "kernel" except this time its called a filter. This sliding window goes over and makes the image fit.

```py
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

