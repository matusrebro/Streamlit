import numpy as np


def moving_average(y, N):
    
    idx_final = y.shape[0]
    yf = np.zeros([idx_final, 1])
    ypf = np.zeros([N, 1])
    ypf[0:N, 0] = y[0:N]
        
    for i in range(0, idx_final):
        if i > 0:
            ypf = np.roll(ypf, 1)
            ypf[0, 0] = y[i]

        yf[i] = 1/(N+1) * (y[i]+np.sum(ypf))
    return yf


# prediction N-steps ahead using ARMA (no input) and recursive least squares algorithm for parameter identification
def arma(y, na, nc, N, fz):
    # data to predict
    # na - order of polynomial of autoregressive part of model
    # nc - order of polynomial of moving average part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    idx_final = y.shape[0]
    y1 = np.zeros([idx_final, 1])
    e1 = np.zeros([idx_final, 1])
    yp = np.zeros([na, 1])
    ep = np.zeros([nc, 1])
    yhat = np.zeros([idx_final, 1])
    theta = np.zeros([idx_final, na + nc])
    P = np.eye(na + nc) * 1e6

    ypredN = np.zeros([idx_final, 1])

    for i in range(0, idx_final):
        y1[i, 0] = y[i]

        for c in range(1, na + 1):
            if i - c >= 0:
                yp[c - 1, 0] = y1[i - c]
            else:
                yp[c - 1, 0] = 0

        for c in range(1, nc + 1):
            if i - c >= 0:
                ep[c - 1, 0] = e1[i - c]
            else:
                ep[c - 1, 0] = 0

        h = np.vstack((-yp, ep))
        y_hat = np.dot(h[:, 0], theta[i - 1, :])
        yhat[i, 0] = y_hat
        e = y1[i, 0] - y_hat
        e1[i, 0] = e
        Y = np.dot(P, h) / (1 + np.dot(np.dot(np.transpose(h), P), h))
        P = (P - np.dot(np.dot(Y, np.transpose(h)), P)) * 1 / fz
        theta[i, :] = theta[i - 1, :] + np.transpose(np.dot(Y, e))
        thetak = np.transpose(theta[i, :])

        yp = np.roll(yp, 1)
        yp[0, 0] = y1[i, 0]
        ep[0, 0] = e
        ypred = np.zeros([N, 1])
        ypred = np.zeros([N, 1])
        for k in range(N):
            if k > 0:
                yp = np.roll(yp, 1)
                yp[0, 0] = ypred[k - 1]
                ep = np.roll(ep, 1)
                ep[0, 0] = 0
            h = np.vstack((-yp, ep))
            ypred[k, 0] = np.dot(h[:, 0], thetak)
        ypredN[i, 0] = ypred[-1, 0]

    return ypredN, theta, thetak, yhat, e1


def arma_prediction(yp, ep, theta, N):
    
    ypred = np.zeros([N, 1])
    for k in range(N):
        if k > 0:
            yp = np.roll(yp, 1)
            yp[0] = ypred[k - 1]
            ep = np.roll(ep, 1)
            ep[0] = 0
        h = np.hstack((-yp, ep))
        ypred[k, 0] = np.dot(h, theta)
    return ypred
