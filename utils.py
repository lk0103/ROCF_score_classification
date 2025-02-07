import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn't work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
# for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import atexit
import os
import time
import functools

np.random.seed(47)


# # Utilities
def onehot_decode(inp):
    return np.argmax(inp, axis=0)


def onehot_encode(idx, num_c):
    if isinstance(idx, int):
        idx = [idx]
    n = len(idx)
    out = np.zeros((num_c, n))
    out[idx, range(n)] = 1
    return np.squeeze(out)


def vector(array, row_vector=False):
    """
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    """
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to measurement
    Returns:
        (*function) New wrapped function with measurement
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('Function [{}] finished in {:.3f} s'.format(func.__name__, elapsed_time))
        return out
    return newfunc


# # Interactive drawing
def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    plt.close()


def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


# # Non-blocking figures still block at end
def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


# # Plotting
palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    use_keypress()
    plt.clf()
    plt.ylim(bottom=0)

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(title)
    plt.show(block=block)


def plot_both_errors(valCEs, valREs, testCE=None, testRE=None, pad=None, block=True,
                     name_file='', show=True):
    fig = plt.figure(2)
    use_keypress()
    plt.clf()

    if pad is None:
        pad = max(len(valCEs), len(valREs))
    else:
        valCEs = np.concatentate((valCEs, [None]*(pad-len(valCEs))))
        valREs = np.concatentate((valREs, [None]*(pad-len(valREs))))
        if testCE is not None:
            testCE = np.concatentate((testCE, [None] * (pad - len(testCE))))
        if testRE is not None:
            testRE = np.concatentate((testRE, [None] * (pad - len(testRE))))

    ax = plt.subplot(2, 1, 1)
    plt.ylim(bottom=0, top=100)
    plt.title('Classification error [%]')
    plt.plot(100*np.array(valCEs), label='val set')

    if testCE is not None:
        plt.plot(100*np.array(testCE), label='test set')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylim(bottom=0, top=1)
    plt.title('Model loss [MSE/sample]')
    plt.plot(valREs, label='val set')

    if testRE is not None:
        plt.plot(testRE, label='test set')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('Error metrics')
    plt.legend()

    if name_file != '':
        plt.savefig(name_file + '.png')

    if show:
        plt.show(block=block)
        plt.close(fig)


def plot_CE(valCEs,testCE=None,  pad=None, block=True,
                     name_file='', show=True):
    fig = plt.figure()
    use_keypress()
    plt.clf()

    if pad is None:
        pad = len(valCEs)
    else:
        valCEs = np.concatentate((valCEs, [None]*(pad-len(valCEs))))
        if testCE is not None:
            testCE = np.concatentate((testCE, [None] * (pad - len(testCE))))

    plt.ylim(bottom=0, top=100)
    plt.title('Classification error [%]')
    plt.plot(100*np.array(valCEs), label='val set')

    if testCE is not None:
        plt.plot(100*np.array(testCE), label='test set')
    plt.legend()


    #plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('Error metrics')
    plt.legend()

    if name_file != '':
        plt.savefig(name_file + '.png')

    if show:
        plt.show(block=block)
        plt.close(fig)


def plot_RE(valREs, testRE=None, pad=None, block=True,
                     name_file='', show=True):
    fig = plt.figure()
    use_keypress()
    plt.clf()

    if pad is None:
        pad = len(valREs)
    else:
        valREs = np.concatentate((valREs, [None]*(pad-len(valREs))))
        if testRE is not None:
            testRE = np.concatentate((testRE, [None] * (pad - len(testRE))))


    plt.ylim(bottom=min(valREs), top=max(valREs))
    plt.title('Model loss [MSE/sample]')
    plt.plot(valREs, label='val set')

    if testRE is not None:
        plt.plot(testRE, label='test set')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('Error metrics')
    plt.legend()

    if name_file != '':
        plt.savefig(name_file + '.png')

    if show:
        plt.show(block=block)
        plt.close(fig)


