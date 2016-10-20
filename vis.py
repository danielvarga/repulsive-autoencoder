from PIL import Image
import numpy as np

def plotImages(data, n_x, n_y, name):
    height, width = data.shape[-2:]
    height_inc = height+1
    width_inc = width+1
    n = len(data)
#    assert n <= n_x*n_y
    if n > n_x*n_y: n = n_x * n_y

    image_data = np.zeros(
        (height_inc * n_y + 1, width_inc * n_x - 1),
        dtype='uint8'
    )
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx].reshape((height, width))
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data)
    img.save(name+".png")
