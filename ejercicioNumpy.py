from cmath import sqrt
from operator import iadd
from turtle import width
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import os
import copy
import math

def sigmoid(z):
    s = 1/ (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
  w = np.zeros((dim, 1))
  b = 0.
  return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    z = w.T
    A = sigmoid(np.dot((w.T), X) + b)

    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1-A))

    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
            "db": db}    
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0 :
            costs.append(cost)

            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
            "b": b}

    return params, costs

def getArrayFromImage(file, targetSize):
    try:
        # open image
        img = Image.open(file)
    except:
        print("Image not found")
        return 0

    # Make image an square
    width, height = img.size
    size = width if width < height else height
    top = (height - size) / 2
    bottom = top + size
    left = (width - size) / 2
    right = left + size
    img = img.crop((left, top, right, bottom))

    # Resize the image
    side = int(math.sqrt(targetSize / 3)) #assuming 3 dimension RGB image
    img = img.resize((side, side))

    # Convert to numpy
    npImg = np.array(img)

    #flaten immage
    npImg = npImg.reshape(1, targetSize)
    return npImg

class Model():

    def __init__(self, modelType):
        self.modelType = modelType


    def fit(self,X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        m = X_train.shape[0]
        self.w, self.b = initialize_with_zeros(m)

        parameters, costs = optimize(self.w, self.b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=print_cost)

        self.w = parameters['w']
        self.b = parameters['b']

        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        self.costs = costs
        self.lr = learning_rate

    def predict(self, X, rounded=True):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))

        try:
            self.w = self.w.reshape(X.shape[0], 1)
            A = sigmoid(np.dot((self.w.T), X) + self.b)

                
            if (rounded):
                for index, _A in enumerate(A[0]):
                        if _A > 0.5:
                            Y_prediction[0, index] = int(1)
                        else:
                            Y_prediction[0, index] = int(0)
            else:
                Y_prediction = A

            return Y_prediction

        except AttributeError:
            print('No existe el atributo self.w')

################################################################################
################################################################################

cwd = os.path.dirname(__file__)
#curDir = os.path.join(curFile)

# Representacion de las etiquetas 0 y 1 respectivamente
classes = ["Galgo afgano","Bedlington"]

trainSet = h5py.File(cwd + '/dogs_train.h5')
testSet = h5py.File(cwd + '/dogs_test.h5')

print(trainSet.keys())
print(testSet.keys())

train_x = np.array(trainSet['dogs_train_x'][:])
train_y = np.array(trainSet['dogs_train_y'][:])
test_x = np.array(testSet['dogs_test_x'][:])
test_y = np.array(testSet['dogs_test_y'][:])

print("\n\n Finding orignal range")
print("Max: ", np.max(test_x))
print("Min: ", np.min(test_x))

train_x = train_x / 255.
test_x = test_x/ 255.

print("\n\n Verifying new range")
print('train_images min =', train_x.min())
print('train_images max =', train_x.max())
print('test_images min =', test_x.min())
print('test_images max =', test_x.max())

n_img, imgW, imgH, n_colors = train_x.shape
train_x = train_x.reshape(n_img, imgW * imgH * n_colors).T

n_img, imgW, imgH, n_colors = test_x.shape
test_x = test_x.reshape(n_img, imgW * imgH * n_colors).T

test_y = np.expand_dims(test_y, 0)
train_y = np.expand_dims(train_y, 0)

print("\n\n Getting number of immages ")
y_train = [list(train_y[0]).count(0), list(train_y[0]).count(1)]
y_test = [list(test_y[0]).count(0), list(test_y[0]).count(1)]
#plt.bar(classes, y_train)
#plt.bar(classes, y_test)
#plt.show()  // uncomment to show plot

print("\n\n Testing sigmoid function, expected: [0.5        0.88079708]")
print("Result:")
print (str(sigmoid(np.array([0,2]))))

print("\n\n Testing propagate function, expected:"
    "\n\tdw = [[0.99845601]"
    "\n\t[2.39507239]]"
    "\n\tdb = 0.001455578136784208"
    "\n\tcost = 5.801545319394553")
print("Result:")

w =  np.array([[1.], [2.]])
b = 2.
X =np.array([[1., 2., -1.], [3., 4., -3.2]])
Y = np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

print("\n\n Testing optimize function, expected:"
    "\n\tdw = [[0.99845601]"
    "\n\t [2.39507239]]"
    "\n\tdb = 0.001455578136784208"
    "\n\tCosts = [array(5.80154532)]")
print("Result:")

params, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print("Costs = " + str(costs))

dogModel = Model("logistic_regression")

num_iterations = 1000
lr = 0.003
print("\n\n Trying out model:")
dogModel.fit(train_x,train_y,test_x,test_y,num_iterations,lr,True)

costs = dogModel.costs
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(dogModel.lr))
plt.show()

print("\n\n Testing model with images:")
test_imgs = np.empty((5, imgW * imgH * n_colors))
test_imgs[0] = getArrayFromImage(cwd + "/g_1.jpg", imgW * imgH * n_colors)
test_imgs[1] = getArrayFromImage(cwd + "/b_1.jpg", imgW * imgH * n_colors)
test_imgs[2] = getArrayFromImage(cwd + "/g_2.jpg", imgW * imgH * n_colors)
test_imgs[3] = getArrayFromImage(cwd + "/b_2.jpg", imgW * imgH * n_colors)
test_imgs[4] = getArrayFromImage(cwd + "/gato.jpg", imgW * imgH * n_colors)
test_imgs = test_imgs.T

y_hat = dogModel.predict(test_imgs, rounded=False)

for index, y in enumerate(y_hat[0]):
    print(f"La predicion de la imagen {index+1} es: {classes[int(y)]} {y}")
    
