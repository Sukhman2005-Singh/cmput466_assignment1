# TODO: Copy from Q2a and/or Q2b as needed
#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    y_hat = np.dot(X, w)
    loss = None
    risk = None

    if y is not None:
        M = X.shape[0]
        loss = np.sum((y_hat - y) ** 2) / (2 * M)

        y_hat_orig = y_hat * std_y + mean_y
        y_orig = y * std_y + mean_y
        risk = np.sum(np.abs(y_hat_orig - y_orig)) / M

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    w = np.zeros([X_train.shape[1], 1])

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    num_batches = int(np.ceil(N_train / batch_size))

    for epoch in range(MaxIter):
        loss_this_epoch = 0
        for b in range(num_batches):
            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            M_batch = X_batch.shape[0]
            grad = np.dot(X_batch.T, (y_hat_batch - y_batch)) / M_batch
            w = w - alpha * (grad + decay * w)

        avg_train_loss = loss_this_epoch / num_batches
        losses_train.append(avg_train_loss)

        _, _, val_risk = predict(X_val, w, y_val)
        risks_val.append(val_risk)

        if val_risk < risk_best:
            risk_best = val_risk
            epoch_best = epoch
            w_best = w.copy()

    return epoch_best, risk_best, w_best, losses_train, risks_val


############################
# Main code starts here
############################
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

X_quad = np.concatenate((X, X ** 2), axis=1)
X_quad = (X_quad - np.mean(X_quad, axis=0)) / np.std(X_quad, axis=0)
X_ = np.concatenate((np.ones([X_quad.shape[0], 1]), X_quad), axis=1)

mean_y = np.mean(y)
std_y = np.std(y)
y = (y - np.mean(y)) / np.std(y)

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

batch_size = 10
MaxIter = 100
decay = 0.01

alpha_list = [0.0001, 0.001, 0.01]
epochs = range(1, MaxIter + 1)

plt.figure()
for a in alpha_list:
    alpha = a
    epoch_best, risk_best, w_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)
    _, _, test_risk = predict(X_test, w_best, y_test)
    print("alpha =", a, "| Best Epoch (0-idx) =", epoch_best, "| Val Risk =", risk_best, "| Test Risk =", test_risk)
    plt.plot(epochs, risks_val, label='alpha=' + str(a))

plt.xlabel('Number of Epochs')
plt.ylabel('Validation Risk')
plt.title('Validation Risk for Different Learning Rates (Q2c)')
plt.legend()
plt.tight_layout()
plt.savefig('Q2c_learning_rates.jpg')