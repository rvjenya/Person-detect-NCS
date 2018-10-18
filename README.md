# Person-detect-NCS

![title](https://cdn-images-1.medium.com/max/1400/1*nd_MjsLUR3yT2YPKUJCptQ.jpeg)

This project based on **ncappzoo** project - by **security-cam**
Now.. it's universavity of classifire - use any classes from 0 to 10 and you will get detect this object real-time.

~~~
0: background
1: aeroplane
2: bicycle
3: bird
4: boat
5: bottle
6: bus
7: car
8: cat
9: chair
10: cow
11: diningtable
12: dog
13: horse
14: motorbike
15: person
16: pottedplant
17: sheep
18: sofa
19: train
20: tvmonitor
~~~

Sample code for a camera classification - built using Intel® Movidius™ Neural Compute Stick (NCS).

## Running this example

~~~
git clone https://github.com/rvjenya/Person-detect-NCS.git
cd Person-detect-NCS
~~~

## Configuring this example

This example grabs camera frames from `/dev/video0` by default; If your system has multiple cameras you can choose the required camera using the `--video` option. Below is an example:

~~~
python3 security-cam.py --video 1
~~~

## ArgumentParser

~~~


 '-g', '--graph', type=str,
                         default='graph',
                         help="Absolute path to the neural network graph file." 
'-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0"

'-l', '--labels', type=str,
                         default='labels.txt',
                         help="Absolute path to labels file." 
'-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." 
'-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." 

'-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" 

'-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." 

~~~

