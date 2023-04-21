To pass this course, I implemented the ROCKET algorithm for 
time series classification, as described in this [paper](https://arxiv.org/abs/1910.13051). 
Although it was a requirement to use R, I found that using R 
was too slow when applying multiple kernels to time series. 
Therefore, I embedded C++ code to speed up the process.