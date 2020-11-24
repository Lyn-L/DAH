# Dense Auto-Encoder Hashing

This is the project of our ACM MM paper (Dense Auto-Encoder Hashing for Robust Cross-Modality Retrieval. ACM MM, 2018.)

Our original version is based on DeepLearning Toolbox on MATLAB platform, but it is convinient for most users. Therefore, we reimplement the method using Pytorch, which can achieve similar results.
The comparison results are shown as below:

|            | Original version | Pytorch version |
| ---------- | ---------------- | --------------- |
| MAP on I2T | 0.3223           | 0.3131          |
| MAP on T2I | 0.7504           | 0.7066          |
| Pre on I2T | 0.2599           | 0.2798          |
| Pre on T2I | 0.7035           | 0,7066          |

Feel free to contact us (lynnliu.xmu at gmail dot com), if you have any problems.

