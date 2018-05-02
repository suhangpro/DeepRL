#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch
import numpy as np
import cv2

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RunningStatsNormalizer(BaseNormalizer):
    def __init__(self, read_only=False):
        super(RunningStatsNormalizer, self).__init__(read_only)
        self.needs_reset = True
        self.read_only = read_only

    def reset(self, x_size):
        self.m = np.zeros(x_size)
        self.v = np.zeros(x_size)
        self.n = 0.0
        self.needs_reset = False

    def state_dict(self):
        return {'m': self.m, 'v': self.v, 'n': self.n}

    def load_state_dict(self, stored):
        self.m = stored['m']
        self.v = stored['v']
        self.n = stored['n']
        self.needs_reset = False

    def __call__(self, x):
        if np.isscalar(x) or len(x.shape) == 1:
            # if dim of x is 1, it can be interpreted as 1 vector entry or batches of scalar entry,
            # fortunately resetting the size to 1 applies to both cases
            if self.needs_reset: self.reset(1)
            return self.nomalize_single(x)
        elif len(x.shape) == 2:
            if self.needs_reset: self.reset(x.shape[1])
            new_x = np.zeros(x.shape)
            for i in range(x.shape[0]):
                new_x[i] = self.nomalize_single(x[i])
            return new_x
        else:
            assert 'Unsupported Shape'

    def nomalize_single(self, x):
        is_scalar = np.isscalar(x)
        if is_scalar:
            x = np.asarray([x])

        if not self.read_only:
            new_m = self.m * (self.n / (self.n + 1)) + x / (self.n + 1)
            self.v = self.v * (self.n / (self.n + 1)) + (x - self.m) * (x - new_m) / (self.n + 1)
            self.m = new_m
            self.n += 1

        std = (self.v + 1e-6) ** .5
        x = (x - self.m) / std
        if is_scalar:
            x = np.asscalar(x)
        return x

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)

class ImagenetNormalizer(BaseNormalizer):
    def __init__(self,
                 is_hwc=False,
                 is_bgr=False,
                 dtype=np.float32,
                 range_max=255,
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225),
                 resize_hw=(224, 224)):
        BaseNormalizer.__init__(self)
        self.range_max = range_max
        self.rgb_mean = np.array(rgb_mean, dtype=dtype).reshape(3, 1, 1)
        self.rgb_std = np.array(rgb_std, dtype=dtype).reshape(3, 1, 1)
        self.is_hwc = is_hwc
        self.is_bgr = is_bgr
        self.dtype = dtype
        self.resize_hw = resize_hw

    def __call__(self, x):
        if not self.is_hwc:
            x = x.transpose(1, 2, 0)
        if self.resize_hw is not None and (x.shape[0] != self.resize_hw[0] or x.shape[1] != self.resize_hw[1]):
            x = cv2.resize(x, self.resize_hw)
            x = np.maximum(0, np.minimum(self.range_max, x))  # resize might result in expanded range
        if self.is_bgr:
            x = x[:, :, -1]
        x = x.astype(self.dtype) * 1.0 / self.range_max
        x = np.divide(x - self.rgb_mean, self.rgb_std)

        if self.is_hwc:
            x = x.transpose(2, 0, 1)
        return x
