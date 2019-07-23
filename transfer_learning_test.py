import unittest
import numpy as np
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

if __name__ == '__main__':
    unittest.main()
