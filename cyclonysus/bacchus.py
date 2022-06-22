"""
Wrappers for dionysus classes
"""
import warnings

import numpy as np
import dionysus as ds

from typing import List, Set, Tuple, Dict, Union
from datatypes import Number, Node, Edge, Vertex


###############################################################################
#     INDEXED SIMPLEX
###############################################################################
class IndexedSimplex(ds.Simplex):
    """
        Extends `ds.Simplex` with filtration index.

        self.data -> self.weight :: simplex.data == edge.weight
    """
    VERBOSE_MODE = False
    PRINT_DECIMALS = 2

    def __init__(self, *args, **kwargs):
        weight = kwargs.get('weight', 0)
        data = kwargs.setdefault('data', weight)
        self.index = kwargs.get('index')

        # if len(args) == 1 and isinstance(args[0], ds.Simplex):
        if len(args) == 1:
            args = args[0] if isinstance(args[0], ds.Simplex) else ds.Simplex(args[0])
            data = args.data

        # if 'data' or 'weight' was passed it overrules simplex.data
        if kwargs['data'] or kwargs.get('weight'):
            data = kwargs['data']

        super().__init__(args, data)
        pass

    def __repr__(self):
        s = self._verbose if IndexedSimplex.VERBOSE_MODE else self._laconic
        return s

    @property
    def _verbose(self):
        weight = self.weight
        index = self.index
        decimals = IndexedSimplex.PRINT_DECIMALS
        return f"<{','.join([str(v) for v in self])}> [{weight :.{decimals}f}:{index}]"

    @property
    def _laconic(self):
        return f"<{','.join([str(v) for v in self])}>"

    @property
    def weight(self):
        return self.data

    @property
    def dim(self):
        return self.dimension()

    pass


###############################################################################
#     PERSISTENCE INTERVALS
###############################################################################
class PersistenceInterval(ds.DiagramPoint):
    """
    Extends [b, d] weight-interval (`ds.DiagramPoint`)
    with filtration indices [birth_idx, death_idx].

    self.data -> self.birth_idx
    """
    VERBOSE_MODE = False
    PRINT_DECIMALS = 2

    def __init__(self, *args, **kwargs):
        """
        :param args:
                * IF len(args) == 1 THEN args[0] is `interval`
                * IF len(args) == 2 THEN args[0:2] is [`birth`, `death`]
        :param kwargs:
                * 'interval', 'birth', 'death' ONLY IF len(args) == 0
                * 'data', 'birth_idx', 'death_idx',
                * 'persistence'
        Note:
            * 'interval' overrules any other options.
            * 'birth_idx' overrules 'data'
        """
        # OPTIONS -------------------------------------------------------
        if args:
            if len(args) == 1:
                kwargs['interval'] = args[0]
            if len(args) == 2:
                kwargs['birth'] = args[0]
                kwargs['death'] = args[1]

        interval = kwargs.get('interval')
        birth = kwargs.get('birth', 0)
        death = kwargs.get('death', 0)
        data = kwargs.get('data', 0)
        persistence = kwargs.get('persistence')
        birth_idx = kwargs.get('birth_idx', self._get_birth_idx(data, persistence))
        death_idx = kwargs.get('death_idx', self._get_death_idx(birth_idx, persistence))

        if interval:
            birth = interval.birth
            death = interval.death
            birth_idx = data = interval.data
            if isinstance(interval, PersistenceInterval):
                birth_idx = interval.birth_idx
                death_idx = interval.death_idx

        # consistency
        data = birth_idx

        # CONSTRUCTOR ---------------------------------------------------
        super().__init__(birth, death)
        if self.birth > self.death:
            # warnings.warn(f"Birth after death: b={self.birth} > d={self.death}")
            raise ValueError(f"Birth after death: b={self.birth} > d={self.death}", self)

        self.data = data
        self.birth_idx = birth_idx
        self.death_idx = death_idx
        pass

    def __eq__(self, arg0):
        # birth-death compare
        bd = super().__eq__(arg0)
        if isinstance(arg0, PersistenceInterval):
            return bd and \
                   self.birth_idx == arg0.birth_idx and \
                   self.death_idx == arg0.death_idx
        return bd

    def __hash__(self):
        return hash((
            self.birth, self.birth_idx,
            self.death, self.death_idx
        ))

    def __repr__(self):
        s = self._verbose if PersistenceInterval.VERBOSE_MODE else self._laconic
        return s

    @property
    def _verbose(self):
        d = PersistenceInterval.PRINT_DECIMALS
        return f"({self.birth:.{d}g},{self.death:.{d}g})" \
               f"<{self.birth_idx},{self.death_idx}>"

    @property
    def _laconic(self):
        d = PersistenceInterval.PRINT_DECIMALS
        return f"({self.birth:.{d}g},{self.death:.{d}g})"

    @staticmethod
    def _get_birth_idx(data, persistence) -> int:
        # return data if persistence else None
        return data

    @staticmethod
    def _get_death_idx(birth_idx, persistence) -> int:
        if PersistenceInterval._is_paired(birth_idx, persistence):
            return persistence.pair(birth_idx)
        else:
            return None

    @staticmethod
    def _is_paired(birth_idx, persistence) -> bool:
        if persistence:
            return persistence.pair(birth_idx) != persistence.unpaired
        else:
            return False

    @property
    def lifetime(self) -> float:
        return self.death - self.birth

    @property
    def infinite(self) -> bool:
        return np.isinf(self.death) or np.isinf(self.birth)

    @property
    def finite(self) -> bool:
        return not self.infinite

    def get_death_interval(self) -> 'PersistenceInterval':
        """
        :return: [0, d]
        """
        return PersistenceInterval(
            birth=0, death=self.death,
            birth_idx=0, death_idx=self.death_idx,
        )

    def get_birth_interval(self) -> 'PersistenceInterval':
        """
        :return: [0, b]
        """
        return PersistenceInterval(
            birth=0, death=self.birth,
            birth_idx=0, death_idx=self.birth_idx,
        )

    pass


###############################################################################
#     SLICING FILTRATION
###############################################################################
class SlicingFiltration(ds.Filtration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # {simplex -> filtration index}:
        self._index: Dict[ds.Simplex, int] = kwargs.get('index', None)
        # {filtration index -> array index}:
        self._array_index: Dict[int, int] = {
            self.index(s): i for i, s in enumerate(self)
        }
        # interval
        self._interval = None

        # TODO: _monotonicity: make it as a test
        pass

    def __repr__(self):
        n = len(self)
        s = "simplices" if n > 1 else "simplex"
        return f"SlicingFiltration with {n} {s} of dimensions {self.dims}"

    def __contains__(self, arg0):
        if isinstance(arg0, ds.Simplex):
            if self._index:
                return arg0 in self._index
            return super().__contains__(arg0)
        if isinstance(arg0, (list, tuple)):
            return super().__contains__(ds.Simplex(arg0))
        return False

    def __getitem__(self, idx) -> ds.Simplex:
        """
        :param idx: filtration index, min=`birth_idx`, max=`death_idx`
                    doesn't handle idx==-1
        :return: self[idx]
        """
        i = self._array_index[idx]
        return super().__getitem__(i)

    def get(self, i) -> ds.Simplex:
        """
        :param i: array idx, min=`0`, max=`len(self)`,
                    handles idx==-1
        :return: `i`-th element of the underlying array
        """
        i = len(self) + i if i < 0 else i
        return super().__getitem__(i)

    # def get_indexed_simplex(self, idx):
    #     s = self[idx]
    #     return self.as_indexed_simplex(s)
    #
    # def as_indexed_simplex(self, s: Union[ds.Simplex, Tuple[int, ...], List[int]]):
    #     if not isinstance(s, ds.Simplex):
    #         s = ds.Simplex(s)
    #     s_idx = self.index(s)
    #     return IndexedSimplex(self[s_idx], index=s_idx)

    def to_list(self) -> List[IndexedSimplex]:
        return [IndexedSimplex(s, index=self.index(s)) for s in self]

    def to_set(self) -> Set[IndexedSimplex]:
        return {IndexedSimplex(s, index=self.index(s)) for s in self}

    @property
    def dims(self) -> Tuple[int, ...]:
        return sorted(list(set([s.dimension() for s in self])))

    @property
    def interval(self) -> PersistenceInterval:
        first, last = self.first, self.last
        return PersistenceInterval(
            birth=first.data, death=last.data,
            birth_idx=first.index,
            death_idx=last.index,
            data=first.index
        )

    @property
    def first(self) -> IndexedSimplex:
        s_idx = self.index(self.get(0))
        return IndexedSimplex(self[s_idx], index=s_idx)

    @property
    def last(self) -> IndexedSimplex:
        s = self.get(-1)
        s_idx = self.index(s)
        return IndexedSimplex(self[s_idx], index=s_idx)

    # @overrides('index', check_signature=False)
    def index(self, s):
        return super().index(s) if not self._index else self._index[s]

    def _index_dict(self) -> Dict[ds.Simplex, int]:
        return {s: self.index(s) for s in self}

    # @overrides('append', check_signature=False)
    def append(self, s):
        super().append(s)
        s_idx = self.index(s)
        self._array_index[s_idx] = len(self) - 1
        self._interval = None
        pass

    def pop(self, n=1, drop_index=False, popped=None) -> 'SlicingFiltration':
        """Remove last `n` simplices"""
        n = n if n <= len(self) else len(self)
        index = self._index_dict() if not drop_index else None
        filtration = list(self)
        if popped is not None:
            popped.extend(filtration[-n:])
        return SlicingFiltration(filtration[:-n], index=index)

    def by_dims(self, dims, drop_index=False, split=False) -> 'SlicingFiltration':
        index = self._index_dict() if not drop_index else None
        dims = [dims] if not isinstance(dims, (list, tuple)) else dims
        if split:
            filtered = [[s for s in self if s.dimension() == d] for d in dims]
            return tuple([
                SlicingFiltration(f, index=index) for f in filtered
            ])
        else:
            filtered = [s for s in self if s.dimension() in dims]
            return SlicingFiltration(filtered, index=index)
        pass

    def by_pers_interval(self, interval: PersistenceInterval, drop_index=False) -> 'SlicingFiltration':
        if not isinstance(interval, PersistenceInterval):
            raise TypeError(
                f"PersInterval is expected, '{type(interval)}' found instead."
            )
        return self.by_index_interval(interval.birth_idx, interval.death_idx, drop_index)

    def by_index_interval(
            self,
            birth_idx=None,
            death_idx=None,
            drop_index=False
    ) -> 'SlicingFiltration':
        first_idx, last_idx = self.interval.birth_idx, self.interval.death_idx
        birth_idx = first_idx if birth_idx is None else birth_idx
        death_idx = last_idx if death_idx is None else death_idx
        index = self._index_dict() if not drop_index else None

        if not (first_idx <= birth_idx <= death_idx <= last_idx):
            warnings.warn(
                f"Expected Index: "
                f"{first_idx} <= {birth_idx} <= {death_idx} <= {last_idx}"
            )
            return SlicingFiltration([], index=index)

        filtered = [
            s for s in self
            if birth_idx <= self.index(s) <= death_idx
        ]
        return SlicingFiltration(filtered, index=index)

    def by_time_interval(self, birth, death, drop_index=False) -> 'SlicingFiltration':
        m, M = self.first.data, self.last.data
        index = self._index_dict() if not drop_index else None
        if not (m <= birth <= death <= M):
            warnings.warn(
                f"Expected Time: {m} <= {birth} <= {death} <= {M}"
            )
            return SlicingFiltration([], index=index)

        filtered = [
            s for s in self
            if birth <= s.data <= death
        ]
        return SlicingFiltration(filtered, index=index)

    pass

