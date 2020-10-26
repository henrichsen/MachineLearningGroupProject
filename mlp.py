import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.


class BPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, momentum=0, shuffle=True, hidden_layer_widths=None, num_outs=2):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.num_outs = num_outs

        self.mse = []
        self.v_mse = []
        self.t_mse = []
        self.v_acc = []

    def get_MSE(self, x, y):
        return np.mean((x - y) ** 2)

    def get_stuff_for_graph(self, X, y, v_X, v_y, t_X, t_y):
        self.mse.append(self.get_MSE(self.predict(X), y))
        self.v_mse.append(self.get_MSE(self.predict(v_X), v_y))
        self.t_mse.append(self.get_MSE(self.predict(t_X), t_y))
        # self.v_acc.append(self.score(v_X, v_y))
        return

    def sigmoid(self, value):
        return 1 / (1 + (np.math.e ** (-1 * value)))

    def calculateOutputs(self, x, hiddens, outputs):
        prev_layer = x
        h_index = 0
        w_index = 0
        combined = np.append(self.hidden_layer_widths, len(outputs))
        for i in range(len(combined)):
            start_h_index = h_index
            start_start_w_index = w_index
            for j in range(combined[i]):
                increment = combined[i]
                start_w_index = w_index
                total = 0
                for k in range(len(prev_layer)):
                    total += prev_layer[k] * self.weights[w_index]
                    w_index += increment
                # add in that bias boi
                total += 1 * self.weights[w_index]
                if i == len(combined) - 1:
                    outputs[j] = self.sigmoid(total)
                else:
                    hiddens[h_index] = self.sigmoid(total)
                    h_index += 1
                w_index = start_w_index + 1
            w_index = start_start_w_index + (len(prev_layer) + 1) * combined[i]
            prev_layer = hiddens[start_h_index:h_index]
        # print("hiddens", hiddens)
        return hiddens, outputs

    def updateWeights(self, x, targets, hiddens, outputs):
        # first get errors for output nodes
        # targets = np.array([0, 1]) if target == 1 else np.array([1, 0])
        o_errors = np.zeros(len(outputs))
        for i in range(len(outputs)):
            o_errors[i] = (targets[i] - outputs[i]) * outputs[i] * (1 - outputs[i])

        # next is errors for all hidden nodes
        h_errors = np.zeros(len(hiddens))
        k_layer_errors = np.copy(o_errors)
        h_index = len(hiddens)
        w_index = len(self.weights) - ((self.hidden_layer_widths[len(self.hidden_layer_widths) - 1] + 1) * len(outputs))  # + 1 in there is for bias
        for i in range(len(self.hidden_layer_widths) - 1, -1, -1):
            start_w_index = w_index
            start_h_index = h_index - self.hidden_layer_widths[i]
            j_layer = hiddens[start_h_index:h_index]
            h_index = start_h_index
            for j in range(len(j_layer)):
                total = 0
                for k in range(len(k_layer_errors)):
                    total += k_layer_errors[k] * self.weights[w_index] * hiddens[h_index] * (1 - hiddens[h_index])
                    w_index += 1
                h_errors[h_index] = total
                h_index += 1
            h_index = start_h_index
            k_layer_errors = h_errors[start_h_index:start_h_index + self.hidden_layer_widths[i]]
            if i - 1 != -1:
                w_index = start_w_index - (self.hidden_layer_widths[i - 1] + 1) * self.hidden_layer_widths[i]  # + 1 for dat bias
        print("h_errors", h_errors)
        print("o_errors", o_errors)
        print("Descending gradient...")
        print("\n")

        k_errors = o_errors
        h_index = len(hiddens)
        w_index = len(self.weights) - ((self.hidden_layer_widths[len(self.hidden_layer_widths) - 1] + 1) * len(outputs))  # + 1 in there is for bias
        for i in range(len(self.hidden_layer_widths) - 1, -1, -1):
            start_w_index = w_index
            start_h_index = h_index - self.hidden_layer_widths[i]
            j_outputs = hiddens[start_h_index:h_index]
            h_index = start_h_index
            for j in range(len(j_outputs)):
                for k in range(len(k_errors)):
                    self.dw[w_index] = self.lr * k_errors[k] * j_outputs[j] + self.momentum * self.dw[w_index]
                    w_index += 1
            for k in range(len(k_errors)):
                self.dw[w_index] = self.lr * k_errors[k] * 1 + self.momentum * self.dw[w_index]
                w_index += 1
            k_errors = h_errors[start_h_index:start_h_index + self.hidden_layer_widths[i]]
            if i - 1 != -1:
                w_index = start_w_index - (self.hidden_layer_widths[i - 1] + 1) * self.hidden_layer_widths[i]  # + 1 for dat bias

        w_index = 0
        j_outputs = x
        k_errors = h_errors[0:self.hidden_layer_widths[0]]
        for j in range(len(j_outputs)):
            for k in range(len(k_errors)):
                self.dw[w_index] = self.lr * k_errors[k] * j_outputs[j] + self.momentum * self.dw[w_index]
                w_index += 1
        for k in range(len(k_errors)):
            self.dw[w_index] = self.lr * k_errors[k] * 1 + self.momentum * self.dw[w_index]
            w_index += 1

        for i in range(len(self.weights)):
            self.weights[i] += self.dw[i]

    def fit(self, X, y, v_X, v_y, t_X, t_y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        if not self.hidden_layer_widths:
            self.hidden_layer_widths = [X.shape[1] * 2]

        self.initial_weights = self.initialize_weights(X) if not initial_weights else initial_weights
        self.weights = self.initial_weights.copy()
        self.dw = np.zeros(len(self.weights))

        hidden_layer_size = 0
        for i in range(len(self.hidden_layer_widths)):
            hidden_layer_size += self.hidden_layer_widths[i]
        self.hidden_layer_values = np.zeros(hidden_layer_size)
        print(self.hidden_layer_values)

        self.output_values = np.zeros(self.num_outs)

        maxAccuracy = 0.
        best_weights = self.weights.copy()
        noImproveCount = 0
        epochs = 0

        while noImproveCount < 500:
            for j in range(X.shape[0]):
                print(self.weights)
                print("Input vector: ", X[j])
                print("Target output: ", y[j])
                print("Forward propagating...")
                self.hidden_layer_values, self.output_values = self.calculateOutputs(X[j], self.hidden_layer_values, self.output_values)
                print("Predicted output: ", self.output_values)
                print("Backward propagating...")
                self.updateWeights(X[j], y[j], self.hidden_layer_values, self.output_values)
            # do accuracy calculations for stopping parameter
            accuracy = float("{:.2f}".format(self.score(v_X, v_y)))
            print("v accuracy: ", accuracy)
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
                best_weights = self.weights.copy()
                noImproveCount = 0
            else:
                noImproveCount += 1
            if self.shuffle:
                X, y = self._shuffle_data(X, y)
            epochs += 1
        self.weights = best_weights
        print("Total epochs: ", epochs)
        self.get_stuff_for_graph(X, y, v_X, v_y, t_X, t_y)

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        predictions = np.zeros((X.shape[0], self.num_outs))
        for j in range(X.shape[0]):
            dump, predictions[j] = self.calculateOutputs(X[j], self.hidden_layer_values, self.output_values)
        return predictions

    def initialize_weights(self, X):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """

        size = 0
        num_prev_layer = X.shape[1] + 1  # starts with the number of input nodes + 1 for bias
        for j in range(len(self.hidden_layer_widths)):
            size += num_prev_layer * self.hidden_layer_widths[j]
            num_prev_layer = self.hidden_layer_widths[j] + 1  # + 1 for the bias
        size += num_prev_layer * self.num_outs
        random_weights = np.random.uniform(low=-1, high=1, size=size)
        return np.array(random_weights)

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        predictions = self.predict(X)
        same_count = 0
        for i in range(y.shape[0]):
            if y[i][0] + .05 > predictions[i][0] > y[i][0] - .05:
                if y[i][1] + .05 > predictions[i][1] > y[i][1] - .05:
                    same_count += 1
        return same_count / y.shape[0]

    def train_test_split(self, X, y):
        X, y = self._shuffle_data(X, y)
        split = int(X.shape[0] * .80)
        return X[0:split, :], y[0:split], X[split:, :], y[split:]

    def get_validation_set(self, X, y):
        X, y = self._shuffle_data(X, y)
        split = int(X.shape[0] * .80)
        return X[0:split, :], y[0:split], X[split:, :], y[split:]

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """

        perm = np.random.permutation(range(len(y)))
        return X[perm], y[perm]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
