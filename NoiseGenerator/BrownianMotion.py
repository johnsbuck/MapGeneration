import numpy as np


class BrownianMotion(object):

    @classmethod
    def random_walk(cls, delta=2, time=10., num_steps=50, instances=1):
        delta_time = time / num_steps
        x = np.zeros((num_steps, instances))

        r = np.random.normal(size=x.shape + (num_steps,), scale=delta * np.sqrt(delta_time))
        out = np.empty(r.shape)

        np.cumsum(r, axis=-1, out=out)
        return out


x = BrownianMotion.random_walk(instances=2)
print(x.shape)
