import unittest
import numpy as np
import lgmm
from scipy.spatial.distance import cdist

class TestLGMM(unittest.TestCase):

    def test_slgmm_fit(self):
        # generate a simple, one-dimensional dataset generated using two
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) - 1, np.random.randn(100, 1) + 1])
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])

        # initialize an slgmm
        model = lgmm.SLGMM(K = '1 per class')
        # train it
        model.fit(X, y)
        # assert that we found the correct locations
        np.testing.assert_allclose(model._Mus, np.array([[-1., +1.]]), atol=0.5)

        # now try to learn this model without crisp assignments
        model = lgmm.SLGMM(K = 2, max_it = 100)
        # train it
        model.fit(X, y)
        # assert that we found the correct locations for the means
        Mus_expected = np.array([[-1.], [+1.]])
        D = cdist(Mus_expected, model._Mus.T, 'euclidean')
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)

        # generate a more complicated dataset, where there are two Gaussian
        # clusters per class
        X = np.concatenate([
            np.random.randn(100, 2) + np.array([[-1., -1.]]),
            np.random.randn(100, 2) + np.array([[+1., +1.]]),
            np.random.randn(100, 2) + np.array([[-1., +1.]]),
            np.random.randn(100, 2) + np.array([[+1., -1.]])
        ])
        y = np.concatenate([np.ones((200)) * -1, np.ones((200))])

        # initialize an slgmm
        model = lgmm.SLGMM(K = '2 per class')
        # train it
        model.fit(X, y)
        # assert that we found the correct locations for the means
        Mus_expected = np.array([[-1., -1.], [+1., +1.], [-1., +1.], [+1., -1.]])
        D = cdist(Mus_expected[:2, :], model._Mus[:, :2].T, 'euclidean')
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)
        D = cdist(Mus_expected[2:, :], model._Mus[:, 2:].T, 'euclidean')
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)

if __name__ == '__main__':
    unittest.main()
