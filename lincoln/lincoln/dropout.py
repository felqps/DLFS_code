import numpy as np

from lincoln import base


class Dropout(base.Operation):
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self) -> np.ndarray:
        if self.inference:
            return self.input_ * self.keep_prob

        self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
        return self.input_ * self.mask

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask
