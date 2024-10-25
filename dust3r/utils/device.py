import tinygrad
import numpy as np

def collate_with_cat(whatever, lists=False):
    if isinstance(whatever, dict):
        return {k: collate_with_cat(vals, lists=lists) for k, vals in whatever.items()}

    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)

        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {k: collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem}

        if isinstance(elem, tinygrad.Tensor):
            print(whatever)
            return listify(whatever) if lists else tinygrad.Tensor.cat(*whatever)
        if isinstance(elem, np.ndarray):
            return listify(whatever) if lists else tinygrad.Tensor.cat(*(tinygrad.Tensor(x) for x in whatever))

        # otherwise, we just chain lists
        return sum(whatever, T())

def listify(elems):
    return [x for e in elems for x in e]
