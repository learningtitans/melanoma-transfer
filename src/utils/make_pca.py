from __future__ import division, print_function

import click
import numpy as np

import data
import util


def compute_mean(files, batch_size=128):
    """Load images in files in batches and compute mean."""
    m = np.zeros(3)
    for i in range(0, len(files), batch_size):
        images = data.load_image(files[i : i + batch_size])
        m += images.sum(axis=(0, 2, 3))
    return (m / len(files)).astype(np.float32)


def std(files, batch_size=128):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print("done with {:>3} / {} images".format(i, len(files)))
        images = np.array(data.load_image(files[i : i + batch_size]),
                          dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)

    print('mean')
    print((s / n).astype(np.float32))
    print('std')
    print(np.sqrt(var))
    #return np.sqrt(var)


@click.command()
@click.option('--directory', default=None)
def main(directory):

    filenames = data.get_image_files(directory)


    bs = 1000
    batches = [filenames[i * bs : (i + 1) * bs]
               for i in range(int(len(filenames) / bs) + 1)]

    # compute mean and std
    std(filenames, bs)

    Us, evs = [], []
    for batch in batches:
        images = np.array([data.load_augment(f, 128, 128) for f in batch])
        X = images.transpose(0, 2, 3, 1).reshape(-1, 3)
        cov = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        ev = np.sqrt(S)
        Us.append(U)
        evs.append(ev)

    print('U')
    print(np.mean(Us, axis=0))
    print('eigenvalues')
    print(np.mean(evs, axis=0))



if __name__ == '__main__':
    main()
