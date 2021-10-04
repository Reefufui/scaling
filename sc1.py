import matplotlib
import numpy as np
import itertools

from skimage import io

def get_y_img(raw_img):
    h, w, c = raw_img.shape
    y_img = np.zeros((h, w), dtype='float64')
    
    xs, ys = np.arange(h), np.arange(w)
    for x, y in itertools.product(xs, ys):
        y_img[x, y] = np.dot(raw_img[x, y], np.array([0.229, 0.587, 0.111]))
    
    return y_img

def get_e_img(y_img):
    h, w = y_img.shape
    xs, ys = np.arange(1, w + 1), np.arange(1, h + 1)
    e_img = np.zeros((h + 2, w + 2), dtype='float64')
    img = np.pad(y_img, pad_width=1, mode='edge')
    
    for x, y in itertools.product(xs, ys):
        dx = img[y + 1, x] - img[y - 1, x]
        dy = img[y, x + 1] - img[y, x - 1]
        e_img[y, x] = np.sqrt(dx*dx + dy*dy)
    
    return e_img[1:-1, 1:-1]

def get_seam_matrix(y_img, mode='horizontal shrink'):
    h, w = y_img.shape
    xs, ys = np.arange(1, w + 1), np.arange(1, h + 1)
    result = np.pad(y_img, pad_width=1, mode='constant', constant_values=(10000000))
    
    mode = mode.split()[0]
    strides = np.zeros((h, w), dtype = "int")
    
    if mode == 'horizontal':
        ys = ys[1:]
        for y in ys:
            for x in xs:
                stride = np.argmin(result[y - 1, x - 1 : x + 2]) - 1
                result[y, x] += result[y - 1, x + stride]
                strides[y - 1, x - 1] = stride
    elif mode == 'vertical':
        xs = xs[1:]
        for x in xs:
            for y in ys:
                stride = np.argmin(result[y - 1 : y + 2, x - 1]) - 1
                result[y, x] += result[y + stride, x - 1]
                strides[y - 1, x - 1] = stride

    else:
        raise ValueError
    
    result = result[1:-1, 1:-1]

    return result, strides

def get_seam_mask(seam_matrix, strides, mode='horizontal shrink'):
    h, w = seam_matrix.shape
    seam_mask = np.zeros((h, w), dtype='int')
    
    index_of_minimal = lambda arr : np.where(arr == np.amin(arr))[0][0]
    mode = mode.split()[0]
    x, y = w - 1, h - 1
    
    if mode == 'horizontal':
        x = index_of_minimal(seam_matrix[y])
        seam_mask[y, x] = 1
        while y:
            y -= 1
            x += strides[y, x]
            seam_mask[y, x] = 1

    elif mode == 'vertical':
        y = index_of_minimal(seam_matrix[..., x])
        seam_mask[y, x] = 1
        while x:
            x -= 1
            y += strides[y, x]
            seam_mask[y, x] = 1

    else:
        raise ValueError
    
    return seam_mask

def shrink(seam_mask, img, mode):
    mode = mode.split()[0]
    h, w, c = img.shape
    
    xs, ys = np.arange(w), np.arange(h)
    result_shape = (h, w - 1, c) if mode == 'horizontal' else (w, h - 1, c)
    result = np.zeros(result_shape, dtype='int')
    
    if mode == 'vertical':
        xs, ys = np.arange(h), np.arange(w)
        img = (np.transpose(img[..., 0]), np.transpose(img[..., 1]), np.transpose(img[..., 2]))
        img = np.dstack(img)
        seam_mask = np.transpose(seam_mask)
    
    for y, x in itertools.product(ys, xs):
        if seam_mask[y][x] == 1:
            result[y][:x] = img[y][:x]
            result[y][x:] = img[y][x+1:]
    
    if mode == 'vertical':
        result = (np.transpose(result[..., 0]),
                  np.transpose(result[..., 1]),
                  np.transpose(result[..., 2]))
        result = np.dstack(result)
        seam_mask = np.transpose(seam_mask)
            
    return result

def seam_carve(img, mode, mask=None):
    y_img = get_y_img(img)
    e_img = get_e_img(y_img)
    seam_matrix, strides = get_seam_matrix(e_img, mode)
    seam_mask = get_seam_mask(seam_matrix, strides, mode)
    
    result = shrink(seam_mask, img, mode)
            
    
    return result, mask, seam_mask
