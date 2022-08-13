Numpy excercise
============

The purpose of this excercise is to craete an AI model to calsify dog pictures (between **Afghan hound** & **Bedlignton terrier**). All the resources seen in the course has to be applied (to create functions and algorithms to process data, and use it for the trainning)

Code
============

All the code is contained in ejercicioNumpy.py script.

Setup
============

A simple way to setup enviroment and run the script is (below steps were tested in Ubunut):

 1. To create a python virtual enviroment
 ~~~bash
python3 -m venv vritualEnv
~~~
 2. Activate the virtual envirioment
~~~bash
cd vritualEnv/bin
source activate
cd ../..
~~~
 3. Install python requirements inside virtual enviroment
~~~bash
(vritualEnv) pip install -r requirements.txt
~~~

Files
============

Data set of images used for trainning and testing model

- dogs_test.h5
- dogs_train.h5

***
Images not belonging to data set used to try out model after trainning

- b_1.jpg    (Bedlignton terrier)
- b_2.jpg    (Bedlignton terrier)
- g_1.jpg    (Afghan hound)
- g_2.jpg    (Afghan hound)
- gato.jpg   (Not even a dog)

***

How to run it
============
Just checkout this repository, make sure to have your setup ready, and run 
~~~bash
python ejercicioNumpy.py 
~~~
