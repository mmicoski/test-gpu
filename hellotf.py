''' Simple operations that show if tensorflow is really being able to use the GPU 
Based on: https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
'''

print("\nimport tf")
import tensorflow as tf


gpuav = tf.test.is_gpu_available()
gpuname = tf.test.gpu_device_name()
print("===============")
print("GPU available=",gpuav)
print("GPU name=<%s>"%str(gpuname))
print("===============")


print("\ntf.constant")
hello = tf.constant('Hello, TensorFlow!')

print("\ntf.Session")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print("\nsess.run")
print(sess.run(hello))


print("\nwith tf.device('/gpu:0')")
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))