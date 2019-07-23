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
        np.testing.assert_allclose(model._Lambda, np.array([[1.]]), atol=0.5)
        np.testing.assert_allclose(model._Pi, np.array([0.5, 0.5]), atol=0.1)

        # now try to learn this model without crisp assignments
        model = lgmm.SLGMM(K = 2, max_it = 100)
        # train it
        model.fit(X, y)
        # assert that we found the correct locations for the means
        Mus_expected = np.array([[-1.], [+1.]])
        D = cdist(Mus_expected, model._Mus.T, 'euclidean')
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)
        np.testing.assert_allclose(model._Lambda, np.array([[1.]]), atol=1.)
        np.testing.assert_allclose(model._Pi, np.array([0.5, 0.5]), atol=0.1)

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
        np.testing.assert_allclose(model._Lambda, np.array([[1., 0.], [0., 1.]]), atol=0.5)
        np.testing.assert_allclose(model._Pi, np.array([0.25, 0.25, 0.25, 0.25]), atol=0.1)

    def test_predict_proba(self):
        # generate a simple, one-dimensional dataset generated using two
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) * 0.1 - 1, np.random.randn(100, 1) * 0.1 + 1])
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])

        # initialize an slgmm
        model = lgmm.SLGMM(K = '1 per class')
        # train it
        model.fit(X, y)

        # predict
        P = model.predict_proba(X)
        # check that the results are right
        P_expected = np.zeros((200, 2))
        P_expected[:100, 0] = 1.
        P_expected[100:, 1] = 1.
        np.testing.assert_allclose(P, P_expected, atol=0.1)

    def test_predict(self):
        # generate a simple, one-dimensional dataset generated using two
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) * 0.1 - 1, np.random.randn(100, 1) * 0.1 + 1])
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])

        # initialize an slgmm
        model = lgmm.SLGMM(K = '1 per class')
        # train it
        model.fit(X, y)

        # predict
        y_pred = model.predict(X)
        # check that the results are right
        np.testing.assert_array_equal(y_pred, y)

if __name__ == '__main__':
    unittest.main()
