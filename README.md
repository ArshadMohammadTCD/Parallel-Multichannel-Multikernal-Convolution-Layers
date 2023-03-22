## This repo was developed by Arshad Rehman Mohammed and Declan Quinn


# Parallel-Multichannel-Multikernal-Convolution-Layers
Parallelising a aspects of a convolutional neural network (CNN).


Modern deep neural networks (DNNs) are one of the most successful types of artificial intelligence. There are many artificial intelligence competitions each years in areas such as image  processing, and pretty much all the winning entries to these
competitions are DNNs. An important type of DNN that is very successful at image classification and other image processing tasks is the convolutional neural network (CNN).

A CNN consists of a directed acyclic graph of "layers" which are typically selected from a small number of standard layers such as activation layers, pooling layers, fully-connected  layers, and convolutional layers. CNNs require very large
amounts of computation to process and input image, and most of this time is spent in convolution layers.

The convolution layers are similar to, but nonetheless different from, standard two-dimensional convolution. In CNNs the images have multiple  "channels" which might be red, green and blue, or simply notional different pieces of information about a single pixel of the image.  In addition, CNN convolution is always performed with multiple different convolution kernels, each of which has multiple channels. We therefore refer to it as multichannel multikernel convolution.
To write an efficient multichannel multikernel convolution routine you  will need an efficient basic algorithm, but it is also important that you take account of issues such as locality of data access and multiple available processor cores.
Your routine should be written in C/C++, and work on the same data  structures as are used in the sample code. You may consult books, papers and online sources, but ALL SOURCES SHOULD BE ACKNOWLEDGED IN YOUR SUBMISSION. All the submitted code should be your own.

The submission should take the form of:
(1) a working, well-written, commented piece of code, and
(2) a document of 3-4 pages pages describing your parallelization and optimization strategy, and list your execution times on a set of standard input sizes.
 
Your routine should be capable of operating efficiently on a range of matrix sizes and shapes. The range of sizes we will consider are:
image_width: 16..512
image height: 16..512
kernel order: 1, 3, 5, or 7
number of channels: 32..2048 (always powers of 2)
number of kernels: 32..2048 (always powers of 2)
The purpose of this lab is to better understand parallelization and code optimization by making changes to speed up some real code on a real machine. Our target machine is stoker.scss.tcd.ie. This machine has four processors.  Each processor has eight out-of-order pipelined, superscalar cores.  And each core has two-way simultaneous multithreading. The cores support Intel SSE4 vector instructions.

To compile with SSE on stoker you should:
- `#include<x86intrin.h>` in your C/C++ code
- compile your program with: `gcc -O3 -msse4 file.c`
To compile with OpenMP, add the flag `-fopenmp`.

Note that this assignment should be completed in pairs of students. The idea of working in pairs is that students might have the opportunity to bounce ideas and plans off another person and share the work, while avoiding the worst problems of group projects where just one or two people do most of the work. The ideal number of students in a pair is two, but pairs with one, two or three students are acceptable.
Every student in the pair should submit the same identical files to Blackboard as their solution. Please make sure that every source file and document that you submit for the lab should have the names of all students who are in the pair displayed prominently at the start of the document (in a comment in source files). When I look at each submitted file, it should be easy for me to find who is the other member of the pair.
