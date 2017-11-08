#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    #print "params", params
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    print "Dx",Dx,"H",H,"Dy",Dy
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    print "W2 shape",W2.shape,"b2 shape",b2.shape
    ### YOUR CODE HERE: forward propagation
    print "data shape",data.shape, "W1 shaep", W1.shape, "b1 shape", b1.shape
    h1 = np.matmul(data, W1) + b1
    a = sigmoid(h1)
    h2 = np.matmul(a, W2) + b2
    y = softmax(h2)
    cost = -np.sum(np.sum(np.multiply(np.log(y), labels), axis=1))
    print "cost",cost
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    print "y shape",y.shape, "labels shaep", labels.shape, "a shape", a.shape
    gradW2 = np.matmul(a.T, (y - labels))
    gradb2 = np.sum((y - labels), axis=0)
    z = np.multiply(np.matmul((y - labels), W2.T),np.multiply(a, (1 - a)))
    gradb1 = np.sum(z, axis=0)
    gradW1 = np.matmul(data.T, z)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
