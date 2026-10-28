#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = np.dot(X, w)
    loss = None
    risk = None

    if y is not None:
        M = X.shape[0]
        loss = np.sum((y_hat - y) ** 2) / (2 * M)

        # Unnormalize predictions and true targets to original price scale
        y_hat_orig = y_hat * std_y + mean_y
        y_orig = y * std_y + mean_y
        risk = np.sum(np.abs(y_hat_orig - y_orig)) / M

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            M_batch = X_batch.shape[0]
            grad = np.dot(X_batch.T, (y_hat_batch - y_batch)) / M_batch
            w = w - alpha * (grad + decay * w)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_train_loss = loss_this_epoch / int(np.ceil(N_train/batch_size))
        losses_train.append(avg_train_loss)
        # 2. Perform validation on the validation set by the risk
        _, _, val_risk = predict(X_val, w, y_val)
        risks_val.append(val_risk)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if val_risk < risk_best:
            risk_best = val_risk
            epoch_best = epoch
            w_best = w.copy()

    # Return some variables as needed
    return epoch_best, risk_best, w_best, losses_train, risks_val


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


# TODO: Your code here
epoch_best, risk_best, w_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)

# Perform test by the weights yielding the best validation performance
_, _, test_risk = predict(X_test, w_best, y_test)

# Report numbers and draw plots as required.
print("Best validation epoch (0-indexed):", epoch_best)
print("Best validation epoch (1-indexed):", epoch_best + 1)
print("Validation performance (risk) in that epoch:", risk_best)
print("Test performance (risk) in that epoch:", test_risk)

epochs = range(1, MaxIter + 1)

plt.figure()
plt.plot(epochs, losses_train)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('Learning Curve of the Training Loss (Q2a)')
plt.tight_layout()
plt.savefig('Q2a_training_loss.jpg')

plt.figure()
plt.plot(epochs, risks_val)
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Risk')
plt.title('Learning Curve of the Validation Risk (Q2a)')
plt.tight_layout()
plt.savefig('Q2a_validation_risk.jpg')