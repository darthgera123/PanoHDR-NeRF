import random
import numpy as np
from PIL import Image
import skimage
import cv2
from imageio import imsave,imread
'''
take float image as input
example:
convert image to HSV space first
--- opencv takes uint8 image: H [0,180], S,V [0,255], donot use this one
--- skimage can process float image:
    H, S, V [0, 1(+)]
    example:
        img = np.random.rand(2,2,3) *2
        Out[14]:
        array([[[1.6629374 , 1.69618671, 0.75216942],
                [0.1510096 , 0.54126582, 1.91480105]],

               [[0.60086363, 0.17315902, 0.40178088],
                [1.0028168 , 0.54994192, 0.28454716]]])
        skimage.color.rgb2hsv(img)
        Out[15]:
        array([[[0.17253685, 0.5565527 , 1.69618671],
                [0.62979003, 0.92113562, 1.91480105]],

               [[0.91091131, 0.71181644, 0.60086363],
                [0.06158197, 0.7162521 , 1.0028168 ]]])
        skimage.color.hsv2rgb(hsv)
        Out[18]:
        array([[[1.6629374 , 1.69618671, 0.75216942],
                [0.1510096 , 0.54126582, 1.91480105]],

               [[0.60086363, 0.17315902, 0.40178088],
                [1.0028168 , 0.54994192, 0.28454716]]])
'''


class AugmentHSV(object):
    def __init__(self, baked_value='small/large', adapt_hue={'src': None, 'dst': None}, adapt_saturation={'src': None, 'dst': None}, 
                 hue_jitter_std=None, sat_jitter_std=None, val_jitter_std=None):
        if baked_value == 'small':
            hue_jitter_std, sat_jitter_std, val_jitter_std = 15/180., 15/180., 0/180.  # int, change to float when apply jitter
        elif baked_value == 'large':
            hue_jitter_std, hue_jitter_std, val_jitter_std = 25/180., 25/180., 0/180.
        self.hue_jitter_std = hue_jitter_std
        self.sat_jitter_std = sat_jitter_std
        self.val_jitter_std = val_jitter_std
        self.adapt_hue = adapt_hue
        self.adapt_saturation = adapt_saturation

    def _generate_random_hsv(self):
        hsv_jitter_value = (np.clip(random.gauss(0, 0.15) * self.hue_jitter_std, -10/255.0, 10/255.0), 
                            np.clip(random.gauss(0, 0.15) * self.sat_jitter_std, -10/255.0, 10/255.0),
                            np.clip(random.gauss(0, 0.15) * self.val_jitter_std, -10/255.0, 10/255.0))
        return hsv_jitter_value

    def apply_random_HSV(self, rgb):
        """
        rgb: numpy float32
        adapt_saturation={'src':0, 'dst':0}, note: put it here to speed up things, in range [0,1]
        return numpy float32 [0,1]
        """
        assert rgb.dtype == 'float32', 'color jitter only support float32 rgb data, input type: %s' % (rgb.dtype)
        # hsvdata = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # hsvdata = np.asarray(Image.fromarray(np.uint8(rgb*255), 'RGB').convert('HSV'))
        hsvdata = skimage.color.rgb2hsv(rgb)
        h, s, v = hsvdata[:, :, 0], hsvdata[:, :, 1], hsvdata[:, :, 2]

        # change hsv
        random_hsv = self._generate_random_hsv()
        # self.adapt_saturation={'src': None, 'dst': None}
        if self.adapt_saturation['src'] is not None and self.adapt_saturation['dst'] is not None:
            s = s + (self.adapt_saturation['dst'] - self.adapt_saturation['src'])/2  # input saturation is float
        if self.adapt_hue['src'] is not None and self.adapt_hue['dst'] is not None:
            h = h + (self.adapt_hue['dst'] - self.adapt_hue['src'])/2  # input saturation is float
        h = h + random_hsv[0]
        v = v + random_hsv[2]

        # make sure HSV are in the correct range
        hmax = 1  # 255.0
        h = h % hmax
        s = np.maximum(0, np.minimum(1, s))
        v = np.maximum(0, v)
        newhsv = np.stack((h, s, v), 2)

        # hsv to rgb
        # rgb = np.array(cv2.cvtColor(newhsv, cv2.COLOR_HSV2RGB))
        # rgb = np.asarray(Image.fromarray(np.uint8(newhsv), 'HSV').convert('RGB')).astype(float)/255.0
        rgb = skimage.color.hsv2rgb(newhsv)
        if np.any(np.isnan(rgb)):
            print('nan value in color jitter')
        # rgb = np.maximum(0, np.minimum(1, rgb))

        return rgb


def compute_mean_hsv(rgb):
    assert rgb.dtype == 'float32', 'color jitter only support float32 rgb data, input type: %s' % (rgb.dtype)
    hsvdata = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsvdata[:, :, 0], hsvdata[:, :, 1], hsvdata[:, :, 2]
    return np.mean(h), np.mean(s), np.mean(v)


def jitter(sigma, seed=None):
    # return random.random() * size - (size / 2)
    if seed:
        random.seed(seed)
    return random.gauss(0, sigma)
    # return size


def jitter3(hue_jitter, sat_jitter, val_jitter):
    return (jitter(hue_jitter), jitter(sat_jitter), jitter(val_jitter))


def hsvColorAdd4D(npRgbColor4D, hue_jitter, sat_jitter, val_jitter):
    rgb = []
    for i in range(npRgbColor4D.shape[0]):
        rgb.append(hsvColorAdd3D(np.squeeze(npRgbColor4D[i, :]), hue_jitter, sat_jitter, val_jitter))
    return np.asarray(rgb)


def hsvColorAdd3D(rgb, hue_jitter, sat_jitter, val_jitter):
    """
    rgb: numpy float32
    hsvJitter: tuple3 [0, 1]
    """
    assert rgb.dtype == 'float32', 'color jitter only support float32 rgb data'
    hsvJitter = jitter3(hue_jitter, sat_jitter, val_jitter)

    hsvdata = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    newhsv = np.array(hsvdata) + np.array(hsvJitter).astype('float32')
    height, width = hsvdata.shape[0:2]
    h = newhsv[:, :, 0].reshape([height, width, 1])
    s = newhsv[:, :, 1].reshape([height, width, 1])
    v = newhsv[:, :, 2].reshape([height, width, 1])

    hmax = 360.0
    h[h > hmax] = h[h > hmax] - hmax
    h[h < 0] = h[h < 0] + hmax
    s = np.maximum(0, np.minimum(1, s))
    # print np.max(s), np.min(s)
    newhsv = np.concatenate((h, s, v), 2)

    rgb = np.array(cv2.cvtColor(newhsv, cv2.COLOR_HSV2RGB))
    if np.any(np.isnan(rgb)):
        print('nan value in color jitter')
    rgb = np.maximum(0, np.minimum(1, rgb))

    return rgb


def hsvColorAdd(rgb, hue_jitter, sat_jitter, val_jitter):
    if len(rgb.shape) == 3:
        newrgb = hsvColorAdd3D(rgb, hue_jitter, sat_jitter, val_jitter)
    elif len(rgb.shape) == 4:
        newrgb = hsvColorAdd4D(rgb, hue_jitter, sat_jitter, val_jitter)
    return newrgb


def colorJitter(rgb, hue_jitter, sat_jitter, val_jitter):
    doFormatCvt = False
    if rgb.dtype == 'uint8':
        doFormatCvt = True
    if doFormatCvt:
        rgb = rgb / 255.

    newrgb = hsvColorAdd(rgb, hue_jitter, sat_jitter, val_jitter)

    if doFormatCvt:
        newrgb = (newrgb * 255).astype('uint8')
    return newrgb


def colorJitterLarge(rgb):
    return colorJitter(rgb, hue_jitter=10, sat_jitter=.1, val_jitter=0)


def colorJitterSmall(rgb):
    return colorJitter(rgb, hue_jitter=5, sat_jitter=.05, val_jitter=0)


if __name__ == '__main__':
    import numpy as np
    # from scipy.misc import imread, imsave
    aa = imread('envmap.jpg') / 255.
    # print(aa.dtype)

    bb = colorJitter(aa.astype('float32'), 5, .05, 10)
    # bb = colorJitterSmall(aa.astype('float32'))
    imsave('envmap-jitter.jpg', (bb*255).astype('uint8'))

    cc = colorJitterLarge(aa.astype('float32'))
    imsave('envmap-jitter_large.jpg', (cc*255).astype('uint8'))