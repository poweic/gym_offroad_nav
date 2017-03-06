import cv2
import numpy as np

def to_image(R, K, interpolation=cv2.INTER_NEAREST):
    R = normalize(R)
    R = cv2.resize(R, (R.shape[1] * K, R.shape[0] * K), interpolation=interpolation)[..., None]
    R = np.concatenate([R, R, R], axis=2)
    return R

def normalize(x):
    value_range = np.max(x) - np.min(x)
    if value_range != 0:
        x = (x - np.min(x)) / value_range * 255.
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x
