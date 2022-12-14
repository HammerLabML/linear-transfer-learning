import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn_lvq.glvq import GlvqModel
from sklearn_lvq.grlvq import GrlvqModel
from sklearn_lvq.gmlvq import GmlvqModel
from sklearn_lvq.grmlvq import GrmlvqModel
from sklearn_lvq.lgmlvq import LgmlvqModel
import em_transfer_learning.lgmm as lgmm

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

    def test_slgmm_predict_proba(self):
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

    def test_slgmm_predict(self):
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

    def test_slgmm_from_lvq(self):
        # generate a simple, two-dimensional dataset generated using two
        # Gaussians
        X = np.random.randn(200, 2) * 0.2
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])
        X[:, 0] += y

        # train a GLVQ model on it
        model = GlvqModel()
        model.fit(X, y)
        # assess the accuracy
        acc = model.score(X, y)
        self.assertTrue(acc > 0.9)
        # generate a slgmm from it
        slgmm = lgmm.slgmm_from_lvq(model, sigma = 0.01)
        # check the accuracy
        acc_slgmm = model.score(X, y)
        np.testing.assert_allclose(acc_slgmm, acc, atol=0.01)

        # train a GRLVQ model on it
        model = GrlvqModel()
        model.fit(X, y)
        # assess the accuracy
        acc = model.score(X, y)
        self.assertTrue(acc > 0.9)
        # generate a slgmm from it
        slgmm = lgmm.slgmm_from_lvq(model, sigma = 0.01)
        # check the accuracy
        acc_slgmm = model.score(X, y)
        np.testing.assert_allclose(acc_slgmm, acc, atol=0.01)

        # train a GMLVQ model on it
        model = GmlvqModel()
        model.fit(X, y)
        # assess the accuracy
        acc = model.score(X, y)
        self.assertTrue(acc > 0.9)
        # generate a slgmm from it
        slgmm = lgmm.slgmm_from_lvq(model, sigma = 0.01)
        # check the accuracy
        acc_slgmm = model.score(X, y)
        np.testing.assert_allclose(acc_slgmm, acc, atol=0.01)

        # train a GRMLVQ model on it
        model = GrmlvqModel()
        model.fit(X, y)
        # assess the accuracy
        acc = model.score(X, y)
        self.assertTrue(acc > 0.9)
        # generate a slgmm from it
        slgmm = lgmm.slgmm_from_lvq(model, sigma = 0.01)
        # check the accuracy
        acc_slgmm = model.score(X, y)
        np.testing.assert_allclose(acc_slgmm, acc, atol=0.01)


    def test_lgmm_fit(self):
        # generate a two-dimensional data with two classes, where one class is
        # stretched along the x axis and one along the y axis
        X = np.random.randn(200, 2)
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])
        X[y < 0, :] *= np.array([[0.5, 0.2]])
        X[y > 0, :] *= np.array([[0.2, 0.5]])
        X[:, 0] += y

        # initialize a lgmm
        model = lgmm.LGMM(K = '1 per class')
        # train it
        model.fit(X, y)
        # assert that we found the correct locations
        np.testing.assert_allclose(model._Mus, np.array([[-1., 0.], [+1., 0.]]).T, atol=0.5)
        np.testing.assert_allclose(model._Lambdas[:, :, 0], np.array([[4., 0.], [0., 25.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Lambdas[:, :, 1], np.array([[25., 0.], [0., 4.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Pi, np.array([0.5, 0.5]), atol=0.1)

        # now try to learn this model without crisp assignments
        model = lgmm.LGMM(K = 2, max_it = 100)
        # train it
        model.fit(X, y)
        # assert that we found the correct locations for the means
        Mus_expected = np.array([[-1., 0.], [+1., 0.]])
        D = cdist(Mus_expected, model._Mus.T, 'euclidean')
        ks = np.argmin(D, axis=1)
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)
        np.testing.assert_allclose(model._Lambdas[:, :, ks[0]], np.array([[4., 0.], [0., 25.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Lambdas[:, :, ks[1]], np.array([[25., 0.], [0., 4.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Pi, np.array([0.5, 0.5]), atol=0.1)

        # generate a more complicated dataset, where there are two Gaussian
        # clusters per class
        X = np.concatenate([
            np.random.randn(100, 2) * np.array([[0.5, 0.2]]) + np.array([[-1., -1.]]),
            np.random.randn(100, 2) * np.array([[0.2, 0.5]]) + np.array([[+1., +1.]]),
            np.random.randn(100, 2) * np.array([[0.5, 0.2]]) + np.array([[-1., +1.]]),
            np.random.randn(100, 2) * np.array([[0.2, 0.5]]) + np.array([[+1., -1.]])
        ])
        y = np.concatenate([np.ones((200)) * -1, np.ones((200))])

        # initialize an lgmm
        model = lgmm.LGMM(K = '2 per class')
        # train it
        model.fit(X, y)
        # assert that we found the correct locations for the means
        Mus_expected = np.array([[-1., -1.], [+1., +1.], [-1., +1.], [+1., -1.]])
        D = cdist(Mus_expected[:2, :], model._Mus[:, :2].T, 'euclidean')
        ks = np.argmin(D, axis=1)
        np.testing.assert_allclose(model._Lambdas[:, :, ks[0]], np.array([[4., 0.], [0., 25.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Lambdas[:, :, ks[1]], np.array([[25., 0.], [0., 4.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)
        D = cdist(Mus_expected[2:, :], model._Mus[:, 2:].T, 'euclidean')
        np.testing.assert_allclose(np.min(D, axis=1), np.zeros(2), atol=0.5)
        ks = np.argmin(D, axis=1)
        np.testing.assert_allclose(model._Lambdas[:, :, 2 + ks[0]], np.array([[4., 0.], [0., 25.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Lambdas[:, :, 2 + ks[1]], np.array([[25., 0.], [0., 4.]]), atol=2., rtol=0.3)
        np.testing.assert_allclose(model._Pi, np.array([0.25, 0.25, 0.25, 0.25]), atol=0.1)

    def test_lgmm_predict_proba(self):
        # generate a simple, one-dimensional dataset generated using two
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) * 0.1 - 1, np.random.randn(100, 1) * 0.1 + 1])
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])

        # initialize an lgmm
        model = lgmm.LGMM(K = '1 per class')
        # train it
        model.fit(X, y)

        # predict
        P = model.predict_proba(X)
        # check that the results are right
        P_expected = np.zeros((200, 2))
        P_expected[:100, 0] = 1.
        P_expected[100:, 1] = 1.
        np.testing.assert_allclose(P, P_expected, atol=0.1)

    def test_lgmm_predict(self):
        # generate a simple, one-dimensional dataset generated using two
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) * 0.1 - 1, np.random.randn(100, 1) * 0.1 + 1])
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])

        # initialize an lgmm
        model = lgmm.LGMM(K = '1 per class')
        # train it
        model.fit(X, y)

        # predict
        y_pred = model.predict(X)
        # check that the results are right
        np.testing.assert_array_equal(y_pred, y)


    def test_lgmm_from_lvq(self):
        # generate a simple, two-dimensional dataset generated using two
        # Gaussians
        X = np.random.randn(200, 2) * 0.2
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])
        X[:, 0] += y

        # train an LGMLVQ model on it
        model = LgmlvqModel()
        model.fit(X, y)
        # assess the accuracy
        acc = model.score(X, y)
        self.assertTrue(acc > 0.9)
        # generate a lgmm from it
        slgmm = lgmm.lgmm_from_lvq(model, sigma = 0.01)
        # check the accuracy
        acc_slgmm = model.score(X, y)
        np.testing.assert_allclose(acc_slgmm, acc, atol=0.01)

if __name__ == '__main__':
    unittest.main()
