from functools import reduce
from ordered_set import OrderedSet


class PointTree(dict):

    def __init__(self, root, *args, **kwargs):
        super(PointTree, self).__init__(*args, **kwargs)

        self.__dict__ = {root: {}}

    def __getitem__(self, *key):
        return reduce(lambda d, k: d[k], key, self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def tracks(self):

        def flatten(d, base=()):
            for k, v in d.items():
                result = base + (k,)
                if isinstance(v, dict) and len(v) != 0:
                    yield from flatten(v, result)
                else:
                    uniquified_result = tuple(OrderedSet(result))
                    if not len(uniquified_result) > 4:
                        continue
                    yield uniquified_result

        return flatten(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def leaf(self, key_path):

        dict_ = self.__dict__
        for key in key_path:
            dict_ = dict_.setdefault(key, {})
        return dict_

    def append(self, key_path, values):

        for v in values:
            self.leaf(key_path)[v] = {}
