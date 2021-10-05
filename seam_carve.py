import numpy as np
import itertools

def get_y_img(img):
    coefs = np.array([0.299, 0.587, 0.114])
    y_img = img.dot(coefs)

    return y_img.astype('float64')

def get_e_img(y_img, mask):
    h, w = y_img.shape
    xs, ys = np.arange(1, w + 1), np.arange(1, h + 1)
    e_img = np.zeros((h + 2, w + 2), dtype='float64')
    img = np.pad(y_img, pad_width=1, mode='edge')
    
    for x, y in itertools.product(xs, ys):
        dx = img[y + 1, x] - img[y - 1, x]
        dy = img[y, x + 1] - img[y, x - 1]
        e_img[y, x] = np.sqrt(dx*dx + dy*dy)

    e_img = e_img[1:-1, 1:-1]
    e_img += mask * (h * w * 256.0)
    
    return e_img.astype('float64')

def get_seam_matrix(y_img):
    h, w = y_img.shape
    xs, ys = np.arange(1, w + 1), np.arange(2, h + 1)
    result = np.pad(y_img, pad_width=1, mode='constant', constant_values=(np.inf))
    
    strides = np.zeros((h, w), dtype = 'int')
    
    for y, x in itertools.product(ys, xs):
        stride = np.argmin(result[y - 1, x - 1 : x + 2]) - 1
        result[y, x] += result[y - 1, x + stride]
        strides[y - 1, x - 1] = stride
    
    result = result[1:-1, 1:-1]

    return result.astype('float64'), strides

def get_seam_mask(seam_matrix, strides):
    h, w = seam_matrix.shape
    seam_mask = np.zeros_like(seam_matrix)

    index_of_minimal = lambda arr : np.where(arr == np.amin(arr))[0][0]

    y, x = h - 1, index_of_minimal(seam_matrix[h - 1, ...])
    seam_mask[y, x] = 1

    while y:
        x += strides[y, x]
        y -= 1
        seam_mask[y, x] = 1

    return seam_mask.astype('float64')

def shrink(img, mask):
    arr, shifts = get_seam_matrix(get_e_img(get_y_img(img), mask))
    seam_mask = get_seam_mask(arr, shifts)

    result_img  = np.zeros_like(img)[:,:-1,:]
    result_mask = np.zeros_like(mask)[:,:-1]

    y, xs = img.shape[0] - 1, np.arange(img.shape[1])

    while y:
        for x in xs:
            if seam_mask[y, x]:
                result_img[y, :x] = img[y, :x]
                result_img[y, x:] = img[y, x+1:]
                result_mask[y, :x] = mask[y, :x]
                result_mask[y, x:] = mask[y, x+1:]
                y -= 1
                break

    return result_img, result_mask, seam_mask

def expand(img, mask):
    arr, shifts = get_seam_matrix(get_e_img(get_y_img(img), mask))
    seam_mask = get_seam_mask(arr, shifts)

    shape = (img.shape[0], img.shape[1] + 1, img.shape[2])
    result_img  = np.zeros(shape, dtype='float64')
    result_mask = np.zeros((shape[0], shape[1]), dtype='int')

    ys, xs = np.arange(img.shape[0]), np.arange(img.shape[1])
    w = img.shape[1]

    for y, x in itertools.product(ys, xs):
        if seam_mask[y, x]:
            result_img[y, :x+1] = img[y, :x+1]
            result_img[y, x+2:] = img[y, x+1:]

            result_mask[y, :x+1] = mask[y, :x+1]
            result_mask[y, x+2:] = mask[y, x+1:]

            result_mask[y, x] += 1

            if x + 1 == w:
                result_img[y, x+1] = result_img[y,x]
            else:
                result_img[y, x+1] = (result_img[y,x] + result_img[y, x+2]) / 2.0

            break

    return result_img, result_mask, seam_mask

def transpose_img(img):
    img = (np.transpose(img[..., 0]), np.transpose(img[..., 1]), np.transpose(img[..., 2]))
    return np.dstack(img)

def seam_carve(img, mode, mask=None):
    if mask is None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype='int')

    hv, se = mode.split(' ')
    transform = shrink if se == 'shrink' else expand

    if hv == 'horizontal':
        return transform(img, mask)

    elif hv == 'vertical':
        img, new_mask, seam_mask = transform(transpose_img(img), np.transpose(mask))
        return transpose_img(img), np.transpose(new_mask), np.transpose(seam_mask)

    else:
        raise ValueError

