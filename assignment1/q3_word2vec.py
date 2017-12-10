#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid
#from q1_gradcheck1 import gradcheck_naive

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x = x / np.sqrt(np.sum(x**2, axis=1).reshape((x.shape[0], 1)))
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    label = np.zeros(outputVectors.shape[0])
    label[target] = 1
    #print "outputVectors shape", outputVectors.shape,"predicted shape",predicted.shape
    #print "outputVectors", outputVectors
    #print "predicted", predicted
    #print "mat", np.matmul(outputVectors, predicted.reshape(predicted.shape[0],1))
    softmax_predicted = softmax(np.matmul(outputVectors, predicted.reshape(predicted.shape[0],1)).reshape(-1))
    #print "softmax_predicted", softmax_predicted
    #print "softmax_predicted shape",softmax_predicted.shape
    #print "label", label
    cost = -np.dot(np.log(softmax_predicted), label)
    #print "cost", cost
    gradPred = np.matmul((softmax_predicted - label), outputVectors)
    #print "gradPred", gradPred
    #print "label shape", label.shape
    #print "predicted shape",predicted.shape,"softmax_predicted shape",softmax_predicted.shape, "softmax_predicted -label shape",(softmax_predicted - label).shape
    grad = np.matmul(((softmax_predicted - label).reshape(label.shape[0],1)), predicted.reshape((1,predicted.shape[0])))
    ### END YOUR CODE
    #print "grad", grad
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    grad = np.zeros(outputVectors.shape)
    k_indices = indices[1:]
    ### YOUR CODE HERE
    uo = outputVectors[target].reshape(-1)
    vc = predicted
    uk = outputVectors[k_indices]
    cost = -np.log(sigmoid(np.dot(uo,vc))) - np.sum(np.log(sigmoid(-np.matmul(uk,vc))))
    x = sigmoid(np.matmul(-uk,vc))-1
    # grad vc
    gradPred = (sigmoid(np.dot(uo,vc))-1) * uo - np.sum(np.matmul(x.reshape(1,x.shape[0]), uk),axis=0)
    # grad uo
    grad[target] = (sigmoid(np.dot(uo,vc))-1) * vc
    # grad uk
    t = -np.outer((sigmoid(np.matmul(-uk,vc))-1),vc)
    i = 0
    for index in k_indices:
        grad[index] += t[i]
        i += 1
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    current_word_index = tokens[currentWord]
    context_words_indices = [tokens[word] for word in contextWords]
    #print "current_word_index", current_word_index
    #print "context_words_indices", context_words_indices
    for i in context_words_indices:
        #print "i",i
        cost_inner, grad_pred_inner, grad_inner = \
            word2vecCostAndGradient(
                predicted=inputVectors[current_word_index],
                target=i,
                outputVectors=outputVectors,
                dataset=dataset
        )
        #print "grad_pred_inner",grad_pred_inner
        cost += cost_inner
        gradIn[current_word_index,:] += grad_pred_inner
        #print "gradIn",gradIn
        gradOut += grad_inner
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    ### YOUR CODE HERE
    current_word_index = tokens[currentWord]
    context_words_indices = [tokens[word] for word in contextWords]
    vc = np.zeros(inputVectors.shape[1])
    for index in context_words_indices:
        vc += inputVectors[index]

    cost_inner, grad_pred_inner, grad_inner = \
        word2vecCostAndGradient(
            predicted=vc,
            target=current_word_index,
            outputVectors=outputVectors,
            dataset=dataset
        )
    cost += cost_inner
    for i in context_words_indices:
        gradIn[i, :] += grad_pred_inner
    gradOut += grad_inner
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient):
    batchsize = 1
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    #print "wordVector", wordVectors
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        #print "C1",C1
        centerword, context = dataset.getRandomContext(C1)
        #print "centerword", centerword, "context",context
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        #print "gin",gin
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)] # left C words, center, right C words, 1+2*C in total
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
                        word2vecModel=skipgram,
                        tokens=dummy_tokens,
                        wordVectors=vec,
                        dataset=dataset,
                        C=5, # around center word, the num of each side context word
                        word2vecCostAndGradient=softmaxCostAndGradient),
                    dummy_vectors)

    gradcheck_naive(lambda vec:
                    word2vec_sgd_wrapper(
                        word2vecModel=skipgram,
                        tokens=dummy_tokens,
                        wordVectors=vec,
                        dataset=dataset,
                        C=5, # around center word, the num of each side context word
                        word2vecCostAndGradient=negSamplingCostAndGradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #     dummy_vectors)

    # print "\n=== Results ==="
    # print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print skipgram("c", 1, ["a", "b"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
    #     negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
    #     negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
