import numpy as np
import arff
import pprint
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

from mlp import BPClassifier


if __name__ == '__main__':
    """
    fp = open('vowel.arff')
    mat = arff.load(fp)

    matData = np.asarray(mat['data'])
    
    data = matData[:, 0:-1]  # .astype(np.float)
    labels = matData[:, -1].reshape(-1, 1)

    pprint.pprint(data)
    pprint.pprint(labels)
    print(data.shape)
    print(labels.shape)
    print("\n")
    """

    fp = os.getcwd()
    rfp = os.path.join("data", "sync1.mat")  # (11472, 4) or 2 for u_v
    mat = loadmat(rfp)
    print(mat)
    p_act = mat['xk'][:, 0:4]
    p_ref = mat['uk']
    u_v = mat['xk'][:, 6:]
    print("p_act: ", p_act)
    print("p_ref: ", p_ref)
    print("u_v: ", u_v)
    data = np.copy(p_act)
    labels = np.copy(u_v)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - 0) / (400000 - 0)
    print(data)
    print(labels)

    BClass = BPClassifier(lr=0.3, momentum=0.8, shuffle=True, hidden_layer_widths=None, num_outs=2)

    trainData, trainLabels, testData, testLabels = BClass.train_test_split(data, labels)

    trainData, trainLabels, v_trainData, v_trainLabels = BClass.get_validation_set(trainData, trainLabels)

    BClass.fit(trainData, trainLabels, v_trainData, v_trainLabels, testData, testLabels)
    Accuracy = BClass.score(testData, testLabels)
    Accuracy2 = BClass.score(trainData, trainLabels)
    Accuracy3 = BClass.score(v_trainData, v_trainLabels)

    mse = BClass.mse
    v_mse = BClass.v_mse
    t_mse = BClass.t_mse
    #v_acc = BClass.v_acc

    print("Test Accuray = [{:.2f}]".format(Accuracy))
    print("Train Accuray = [{:.2f}]".format(Accuracy2))
    print("Validation Accuray = [{:.2f}]".format(Accuracy3))
    print("Final Weights =", BClass.get_weights())
