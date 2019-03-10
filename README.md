# test-gpu
Verify if tensorflow is using gpu.

I have an ASUS Windows 10 machine with GPU. Recently, I had to reinstall Anaconda (transfer from SSD to HD, to open space). I spent half a day getting it right, and one important step was making sure tensorflow was able to use the machine's GPU. It turns out that the simple "hello world with tensorflow" was not enough, as it gave me false positives in some steps.
I then collected some tips on how to do this, from [this](https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell) site and then coded the module hellotf.py in this project. 
