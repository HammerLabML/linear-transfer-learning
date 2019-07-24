import unittest
import numpy as np
from sklearn_lvq.glvq import GlvqModel
from sklearn_lvq.grlvq import GrlvqModel
from sklearn_lvq.gmlvq import GmlvqModel
from sklearn_lvq.grmlvq import GrmlvqModel
from sklearn_lvq.lgmlvq import LgmlvqModel
import lgmm
import transfer_learning

class TestTransferLearning(unittest.TestCase):

    def test_slgmm_transfer_learning(self):
        # generate a simple, one-dimensional dataset generated using three
        # Gaussians
        X = np.concatenate([np.random.randn(100, 1) * 0.2 - 1, np.random.randn(100, 1) * 0.2, np.random.randn(100, 1) * 0.2 + 1])
        y = np.concatenate([np.ones((100)) * -1, np.zeros((100)), np.ones((100))])

        # initialize an slgmm
        model = lgmm.SLGMM(K = '1 per class')
        # train it
        model.fit(X, y)

        # then create target data with only two classes and swapped locations
        Xtar = np.concatenate([np.random.randn(10, 1) * 0.2 - 1, np.random.randn(10, 1) * 0.2 + 1])
        ytar = np.concatenate([np.ones((10)), np.ones((10)) * -1])

        # initialize a transfer learning model
        tl_model = transfer_learning.SLGMM_transfer_model(model)
        # train it
        tl_model.fit(Xtar, ytar)

        # check that the correct transformation was found
        np.testing.assert_allclose(tl_model._H, np.array([[-1.]]), atol=0.5)

    def test_lvq_transfer_learning(self):
        # generate a simple, two-dimensional dataset generated using three
        # Gaussians
        X = np.random.randn(300, 2) * 0.2
        y = np.concatenate([np.ones((100)) * -1, np.zeros((100)), np.ones((100))])
        X[:, 0] += y

        # then create target data with only two classes and swapped locations
        Xtar = np.random.randn(30, 2) * 0.2
        ytar = np.concatenate([np.ones((10)), np.zeros((10)), np.ones((10)) * -1])
        Xtar[:, 0] -= ytar

        # train a GLVQ model on the source data
        model = GlvqModel()
        model.fit(X, y)
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)
        # assert low accuracy on target data
        self.assertTrue(model.score(Xtar, ytar) < 0.5)
        # perform transfer learning, using only two classes
        tl_model = transfer_learning.LVQ_transfer_model(model)
        tl_model.fit(Xtar[ytar < 0.5, :], ytar[ytar < 0.5])
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)

        # train a GRLVQ model on the source data
        model = GrlvqModel()
        model.fit(X, y)
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)
        # assert low accuracy on target data
        self.assertTrue(model.score(Xtar, ytar) < 0.5)
        # perform transfer learning, using only two classes
        tl_model = transfer_learning.LVQ_transfer_model(model)
        tl_model.fit(Xtar[ytar < 0.5, :], ytar[ytar < 0.5])
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)

        # train a GMLVQ model on the source data
        model = GmlvqModel()
        model.fit(X, y)
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)
        # assert low accuracy on target data
        self.assertTrue(model.score(Xtar, ytar) < 0.5)
        # perform transfer learning, using only two classes
        tl_model = transfer_learning.LVQ_transfer_model(model)
        tl_model.fit(Xtar[ytar < 0.5, :], ytar[ytar < 0.5])
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)

        # train a GRMLVQ model on the source data
        model = GrmlvqModel()
        model.fit(X, y)
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)
        # assert low accuracy on target data
        self.assertTrue(model.score(Xtar, ytar) < 0.5)
        # perform transfer learning, using only two classes
        tl_model = transfer_learning.LVQ_transfer_model(model)
        tl_model.fit(Xtar[ytar < 0.5, :], ytar[ytar < 0.5])
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)


    def test_lgmm_transfer_learning(self):
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
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)

        # now, generate target space data which is rotated by 180°
        Xtar = X * np.array([[-1., -1.]])
        ytar = y
        # assert low accuracy
        self.assertTrue(model.score(Xtar, ytar) < 0.5)

        # perform transfer learning
        tl_model = transfer_learning.LGMM_transfer_model(model)
        tl_model.fit(Xtar, ytar)
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)
        # check the transfer matrix
        np.testing.assert_allclose(tl_model._H, np.array([[-1., 0.], [0., 0.]]), atol=0.5)


    def test_lgmlvq_transfer_learning(self):
        # generate a two-dimensional data with two classes, where one class is
        # stretched along the x axis and one along the y axis
        X = np.random.randn(200, 2)
        y = np.concatenate([np.ones((100)) * -1, np.ones((100))])
        X[y < 0, :] *= np.array([[0.5, 0.2]])
        X[y > 0, :] *= np.array([[0.2, 0.5]])
        X[:, 0] += y

        # train an LGMLVQ model on it
        model = LgmlvqModel()
        # train it
        model.fit(X, y)
        # assert high accuracy
        self.assertTrue(model.score(X, y) > 0.9)

        # now, generate target space data which is rotated by 180°
        Xtar = X * np.array([[-1., -1.]])
        ytar = y
        # assert low accuracy
        self.assertTrue(model.score(Xtar, ytar) < 0.5)

        # perform transfer learning
        tl_model = transfer_learning.Local_LVQ_transfer_model(model)
        tl_model.fit(Xtar, ytar)
        # assert high accuracy on target data
        self.assertTrue(model.score(tl_model.predict(Xtar), ytar) > 0.9)
        # check the transfer matrix
        np.testing.assert_allclose(tl_model._H, np.array([[-1., 0.], [0., 0.]]), atol=0.5)

if __name__ == '__main__':
    unittest.main()
