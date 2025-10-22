"""
pycfx/helpers/visualisation.py
Helpers - visualisation tools, see usage in tutorial notebooks.
"""

from pycfx.counterfactual_explanations.differentiable.losses import OptimisationState

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import torch
from sklearn.neighbors import LocalOutlierFactor

def plot_dataset(X_train, y_train, x_axis_ind=0, y_axis_ind=1, cb=True, faded_background=True):
    ax = plt.subplot(1, 1, 1)

    x_min, x_max = X_train[:, x_axis_ind].min() - 0.5, X_train[:, x_axis_ind].max() + 0.5
    y_min, y_max = X_train[:, y_axis_ind].min() - 0.5, X_train[:, y_axis_ind].max() + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    sc = ax.scatter(X_train[:, x_axis_ind], X_train[:, y_axis_ind], c=y_train, cmap="jet", edgecolors="k", alpha=0.15 if faded_background else 0.9, s=100 if faded_background else 40)
    if cb:
        plt.colorbar(sc)
    return ax

def plot_split_dataset(X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1, cb=True, faded_background=True):
    ax = plot_dataset(X_train, y_train, x_axis_ind, y_axis_ind, cb, faded_background)
    ax.scatter(X_test[:, x_axis_ind], X_test[:, y_axis_ind], c=y_test, cmap="jet", edgecolors="w", alpha=0.2 if faded_background else 0.9, s=100 if faded_background else 40)
    return ax

def plot_decision_boundary(model, X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1, unary=False, feature_to_plot=1, faded_background=True): 
    ax = plot_split_dataset(X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, faded_background=faded_background)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100), np.arange(y_min, y_max, (y_max-y_min)/100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    if not unary:
        prob = tf.nn.softmax(Z)
        Z = prob[:, feature_to_plot].numpy()
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2 if faded_background else 0.5, cmap="jet", zorder=0)

    return ax

def plot_counterfactual(factual, counterfactual, model, X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1, conformal=None, faded_background=True): 
    if conformal is not None:
        ax = plot_conformal_prediction(model, conformal, X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, faded_background=faded_background)
    else:
        ax = plot_decision_boundary(model, X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, faded_background=faded_background)

    x1, y1 = factual[x_axis_ind], factual[y_axis_ind]
    x2, y2 = counterfactual[x_axis_ind], counterfactual[y_axis_ind]

    cmap = plt.cm.get_cmap('jet')
    
    pred_factual = np.argmax(model.predict(factual))
    pred_counterfactual = np.argmax(model.predict(counterfactual))

    ax.plot(x1, y1, "o", color=cmap(pred_factual * 100000), mec="w", mew=1, markersize=20)
    ax.plot(x2, y2, "o", color=cmap(pred_counterfactual * 100000), mec="w", mew=1, markersize=20)

    ax.quiver(x1, y1, (x2-x1), (y2-y1), angles='xy', scale_units='xy', scale=1, edgecolor='w', linewidth=1, zorder=100)
    return ax


def plot_conformal_prediction(model, conformal, X_train, X_calib, y_train, y_calib, x_axis_ind=0, y_axis_ind=1, faded_background=True):
    ax = plot_split_dataset(X_train, X_calib, y_train, y_calib, x_axis_ind, y_axis_ind, cb=False, faded_background=faded_background)
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100), np.arange(y_min, y_max, (y_max-y_min)/100))

    Z = conformal.predict_batch(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(list(map(lambda x: len(x), Z)))
    Z[0] = 1
    Z = Z.reshape(xx.shape)

    sc = ax.contourf(xx, yy, Z, alpha=0.2 if faded_background else 0.5, cmap=plt.cm.get_cmap('Grays', 2), zorder=0)
    ax.contourf(xx, yy, Z, levels=[-0.1, 0.1], colors='red', alpha=0.5, zorder=1)

    cb = plt.colorbar(sc, cmap=plt.cm.get_cmap('Grays', 2))
    cb.set_ticks(np.arange(Z.min(), Z.max() + 1, 1))
    cb.set_ticklabels([str(int(tick)) for tick in np.arange(Z.min(), Z.max() + 1, 1)])

    
    return ax


def plot_conformal_prediction_histogram(model, conformal, X_test, x_axis_ind=0, y_axis_ind=1, faded_background=True):
    Z = conformal.predict_batch(X_test)
    Z = np.array(list(map(lambda x: len(x), Z)))
    plt.hist(Z, bins=np.arange(0, 5, 1), edgecolor='black', align='mid')

    plt.xlabel('Set size')
    plt.ylabel('Frequency')
    plt.title('Conformal Prediction Set Sizes')
    plt.show()

def plot_conformal_prediction_coverage(model, conformal, X_test, y_test, x_axis_ind=0, y_axis_ind=1, faded_background=True):
    Z = conformal.predict_batch(X_test)
    coverage = np.zeros_like(y_test)
    for i in range(len(Z)):
        coverage[i] =  y_test[i] in Z[i]

    print(np.mean(coverage))


def plot_kernel(X_train, X_calib, y_train, y_calib, kernel, bandwidth, centre, x_axis_ind=0, y_axis_ind=1):
    ax = plot_split_dataset(X_train, X_calib, y_train, y_calib, x_axis_ind, y_axis_ind, cb=False)
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = np.array([kernel(np.array([x, y]), centre, bandwidth) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    sc = ax.contourf(xx, yy, Z, cmap="gray", zorder=0)
    plt.colorbar(sc)
    return ax


def plot_loss(model, loss, X_train, X_calib, y_train, y_calib, x_axis_ind=0, y_axis_ind=1, y_target=1):
    ax = plot_split_dataset(X_train, X_calib, y_train, y_calib, x_axis_ind, y_axis_ind, cb=False)
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    points = torch.tensor(list(zip(xx.ravel(), yy.ravel())))
    opt_state = OptimisationState(model, None, None, None, None, None, None, torch.tensor(y_target), 1, 1)
    loss_vals = np.zeros((points.shape[0],))
    preds = torch.tensor(model.predict(np.array(points)))

    with torch.no_grad():
        for i in range(points.shape[0]):
            opt_state.x_enc = points[i]
            opt_state.y_enc = preds[i]
            loss_vals[i] = loss.loss(opt_state)

    Z = loss_vals.reshape(xx.shape)
    
    sc = ax.contourf(xx, yy, Z, cmap="gray", zorder=0)
    plt.colorbar(sc)
    return ax

def plot_loss_with_gradients(model, loss_fn, X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1,
                             step=0.2, scale=50, y_target=1, mul=1):
    ax = plot_split_dataset(X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, cb=False)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max + 1, step), np.arange(y_min, y_max + 1, step))

    opt_state = OptimisationState(model, None, None, None, None, None, None, torch.tensor(y_target, device=model.device), 1, 1)

    Z = []
    grads = []

    for (x, y) in zip(xx.ravel(), yy.ravel()):
        point = torch.tensor([[x, y]], dtype=torch.float32, requires_grad=True, device=model.device)

        logits = model.pytorch_model(point)  

        opt_state.x_enc = point
        opt_state.y_enc = torch.squeeze(logits)
        
        loss = loss_fn.loss(opt_state)
        loss *= mul

        loss.backward(retain_graph=True)
        grad = point.grad.detach().cpu().numpy()[0]
        

        Z.append(loss.item())
        grads.append(-1 * grad)

    Z = np.array(Z).reshape(xx.shape)
    grads = np.array(grads).reshape(xx.shape + (2,))

    sc = ax.contourf(xx, yy, Z, cmap="gray", zorder=0)
    plt.colorbar(sc)

    ax.quiver(xx, yy, grads[:, :, 0], grads[:, :, 1],
              color="red", scale=scale, width=0.003)

    return ax

def plot_lof(n_neighbours, X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1, y_target=None, lof_X=None, lof_y=None):
    ax = plot_split_dataset(X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, cb=False)

    lof = LocalOutlierFactor(n_neighbors=n_neighbours, novelty=True, n_jobs=-1)

    if lof_X is None:
        lof_X = X_train
        lof_y = y_train
    
    if y_target is not None:
        lof_X = lof_X[lof_y == y_target]


    lof.fit(lof_X)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = lof.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    sc = ax.contourf(xx, yy, Z, cmap="gray", zorder=0)
    plt.colorbar(sc)
    return ax


def plot_impl(prop_included, y_target, X_train, X_test, y_train, y_test, x_axis_ind=0, y_axis_ind=1):
    ax = plot_split_dataset(X_train, X_test, y_train, y_test, x_axis_ind, y_axis_ind, cb=False)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    xs = np.c_[xx.ravel(), yy.ravel()]

    impl_scores = np.zeros((xs.shape[0],))
    data_by_class = {}

    for y_i in np.unique(y_test):
        data_ind_class_y = np.where(y_train == y_i)
        data_X_class_y = X_train[data_ind_class_y]
        data_by_class[int(y_i)] = data_X_class_y
        
    for i in range(xs.shape[0]):
        X_counterfactual = xs[i]
        data_class_y = data_by_class[int(y_target)]

        distances = np.linalg.norm(data_class_y - X_counterfactual, axis=1, ord=2)
        distances = np.sort(distances)

        impl_scores[i] = np.mean(distances[:int(len(distances)*prop_included)])

    Z = impl_scores.reshape(xx.shape)
    
    sc = ax.contourf(xx, yy, Z, cmap="gray", zorder=0)
    plt.colorbar(sc)
    return ax