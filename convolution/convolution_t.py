import numpy as np


def conv(X, filters, stride=1, pad=0):
    n, c, h, w = X.shape
    n_f, _, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    print(in_X)
    out = np.zeros((n, n_f, out_h, out_w))
   
    print(np.sum(in_X[0,:,0:3,0:3]*filters[0]))

    for i in range(n): # for each image.
        for c in range(n_f): # for each channel.
            for h in range(out_h): # slide the filter vertically.
                h_start = h * stride
                h_end = h_start + filter_h
                for w in range(out_w): # slide the filter horizontally.
                    w_start = w * stride
                    w_end = w_start + filter_w
                    # Element-wise multiplication.
                    out[i, c, h, w] = np.sum(in_X[i, :, h_start:h_end, w_start:w_end] * filters[c])

    return out


X = np.asarray([
# image 1
[
    [[1, 2, 9, 2, 7],
    [5, 0, 3, 1, 8],
    [4, 1, 3, 0, 6],
    [2, 5, 2, 9, 5],
    [6, 5, 1, 3, 2]],

    [[4, 5, 7, 0, 8],
    [5, 8, 5, 3, 5],
    [4, 2, 1, 6, 5],
    [7, 3, 2, 1, 0],
    [6, 1, 2, 2, 6]],

    [[3, 7, 4, 5, 0],
    [5, 4, 6, 8, 9],
    [6, 1, 9, 1, 6],
    [9, 3, 0, 2, 4],
    [1, 2, 5, 5, 2]]
],
# image 2
[
    [[7, 2, 1, 4, 2],
    [5, 4, 6, 5, 0],
    [1, 2, 4, 2, 8],
    [5, 9, 0, 5, 1],
    [7, 6, 2, 4, 6]],

    [[5, 4, 2, 5, 7],
    [6, 1, 4, 0, 5],
    [8, 9, 4, 7, 6],
    [4, 5, 5, 6, 7],
    [1, 2, 7, 4, 1]],

    [[7, 4, 8, 9, 7],
    [5, 5, 8, 1, 4],
    [3, 2, 2, 5, 2],
    [1, 0, 3, 7, 6],
    [4, 5, 4, 5, 5]]
]
])
print('Images:', X.shape)

filters = np.asarray([
# kernel 1
[
    [[1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]],

    [[3, 1, 3],
    [1, 3, 1],
    [3, 1, 3]],

    [[1, 2, 1],
    [2, 2, 2],
    [1, 2, 1]]
],
# kernel 2
[
    [[5, 1, 5],
    [2, 1, 2],
    [5, 1, 5]],

    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[2, 0, 2],
    [0, 2, 0],
    [2, 0, 2]],
],
# kernel 3
[
    [[5, 1, 5],
    [2, 1, 2],
    [5, 1, 5]],

    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[2, 0, 2],
    [0, 2, 0],
    [2, 0, 2]],
]
])
print('Filters:', filters.shape)

out = conv(X, filters, stride=2, pad=0)
print('Output:', out.shape)
print(out)