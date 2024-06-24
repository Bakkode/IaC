# Author: Andika Ageng P.

import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
