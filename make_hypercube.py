import copy
import numpy as np
from itertools import cycle


class NumpyRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for numpy based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self.prev_random_state)


def make_hypercube(nsamples, ninformative, nconstant=0, nredundant=0, nrepeated=0,
                   flipy=0.01, std=0.25, sep=2., random_seed=42, splits=(80, 10, 10)):

    '''

    Generates an (approximately) balanced dataset based on the vertices of an ninformative dimensional hyeprcube.
    The data is split into two classes, the vertices of the cube are assigned to one class each. No two neighboring
    hypercubes are of the same class, thus yielding a dataset where for each sample each dimension has to be considered
    in order to be classified correctly.
    Data is stacked according to [ninformative, nrepeated, nredundant, nconstant]

    :param nsamples: total number of samples generated
    :param ninformative: number of informative dims, ie the dimensionality of the hypercube
    :param nconstant: number of constant dims appended to informative dims
    :param nredundant: number of dims generated via linear combination of informative dims
    :param nrepeated: number of dims copied from informative dims. dims are copied sequentially and circular (if nrepeated > ninformative)
    :param flipy: % of labels that get flipped (ie label noise)
    :param std: standard deviation used for sampling gaussian blobs
    :param sep: multiplied to cluster centers to move them closer together/ further apart
    :param random_seed:
    :param splits: splits% for dividing data into train/test/validation sets
    :return: dictionary containing all input parameters, data split in train/test/validation, and a few more things
    '''

    nfeat = ninformative + nconstant + nredundant + nrepeated
    hypercube_dataset = {
        'n_samples': nsamples,
        'n_features': nfeat,
        'n_informative': ninformative,
        'n_constant': nconstant,
        'n_redundant': nredundant,
        'n_repeated': nrepeated,
        'n_classes': 2,
        'n_clusters_per_class': int(2**ninformative/2),
        'flipy': flipy,
        'std': std,
        'sep': sep,
        'random_seed': random_seed,
        'mixing_coefficients': None,
    }

    with NumpyRandomSeed(random_seed):
        nclusters = 2 ** ninformative
        vertices = np.vstack([np.array(list(np.binary_repr(n, width=ninformative)), dtype=float) for n in range(nclusters)])
        samples_per_cluster = int(nsamples / nclusters)

        X_informative, Y = [], []
        for vertex in vertices:
            # sum(vertex)=0 or sum(vertices) is even -> label = 0;
            # sum(vertices) is uneven -> label = 1
            _label = np.array([0 if sum(vertex) % 2 == 0 else 1] * samples_per_cluster)
            _data = np.random.normal(0, scale=std, size=(samples_per_cluster, ninformative))

            vertex -= 0.5  # move vertices from origin of the space
            if sep > 0:
                vertex *= sep  # separation is wrt. vertices, not the minimal distance between two adjacent clusters
            _data += vertex.reshape(1, -1)
            X_informative.append(_data)
            Y.append(_label)

        X_informative = np.vstack(X_informative)
        Y = np.hstack(Y)

        nsamples = len(X_informative)
        if flipy > 0:
            flips = np.random.choice(nsamples, max(1, int(nsamples*flipy)))
            for f in flips:
                Y[f] = not Y[f]

        # make a copy to which we stack all other dimensions
        X_full = copy.deepcopy(X_informative)

        if nrepeated > 0:
            _rep = []
            for i in cycle(np.arange(ninformative)):
                if len(_rep) >= nrepeated:
                    break
                _rep.append(X_informative[:, i])
            _rep = np.vstack(_rep).T
            X_full = np.hstack((X_full, _rep))

        if nredundant > 0:
            mixing_coefficients = np.random.uniform(0, 1, size=(ninformative, nredundant))
            mixing_coefficients = mixing_coefficients / np.sum(mixing_coefficients, 0).T
            correlated = X_informative @ mixing_coefficients
            X_full = np.hstack((X_full, correlated))
            hypercube_dataset['mixing_coefficients'] = mixing_coefficients

        if nconstant > 0:
            consts = np.random.normal(0, scale=std, size=(nconstant))
            consts = np.repeat(consts.reshape(1, -1), nsamples, 0)
            X_full = np.hstack((X_full, consts))

        # def cutoff points for train/test/val
        splits = np.array(splits)
        splits = splits / np.sum(splits)
        split_tr = int(np.ceil(splits[0] * nsamples))
        split_te = split_tr + int(np.ceil(splits[1] * nsamples))
        # split_val = split_tr + split_te + int(np.ceil(splits[2] * nsamples))

        # shuffle idxs
        idxs = np.arange(nsamples)
        np.random.shuffle(idxs)

        # split data
        X_tr, X_te, X_val = X_full[idxs[:split_tr]], X_full[idxs[split_tr:split_te]], X_full[idxs[split_te:]]
        Y_tr, Y_te, Y_val = Y[idxs[:split_tr]], Y[idxs[split_tr:split_te]], Y[idxs[split_te:]]

        # package
        hypercube_dataset.update({
            "X_tr": X_tr,
            "X_te": X_te,
            "X_val": X_val,
            "Y_tr": Y_tr,
            "Y_te": Y_te,
            "Y_val": Y_val
        })

        return hypercube_dataset

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = make_hypercube(400, 2, nredundant=1)
    X_tr, Y_tr, X_te, Y_te = dataset['X_tr'], dataset['Y_tr'], dataset['X_te'], dataset['Y_te']
    print(f"redundant dims mixed using \n{dataset['mixing_coefficients']}")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = np.array(['xkcd:cerulean', 'xkcd:goldenrod'])  # https://xkcd.com/color/rgb/
    ax.scatter(
        X_tr[:, 0], X_tr[:, 1], X_tr[:, 2], c=colors[Y_tr]
    )
    ax.set_xlabel('informative 1')
    ax.set_ylabel('informative 2')
    ax.set_zlabel('redundant')
    plt.show()
