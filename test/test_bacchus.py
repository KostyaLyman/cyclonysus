import pytest
import numpy as np
import dionysus as ds
from cyclonysus import bacchus as bs


class TestIndexedSimplex:
    def test__from_simplex(self):
        s = ds.Simplex((1, 2, 3))
        assert s is not None

        # test: from simplex
        idx_s = bs.IndexedSimplex(s)
        assert idx_s is not None
        assert idx_s == s
        assert idx_s.data == s.data
        assert idx_s.weight == s.data
        assert idx_s.dim == s.dimension()
        assert idx_s.index is None

        # test: from indexed simplex
        idx_s = bs.IndexedSimplex(idx_s)
        assert idx_s is not None
        assert idx_s == s
        assert idx_s.data == s.data
        assert idx_s.weight == s.data
        assert idx_s.dim == s.dimension()
        assert idx_s.index is None

        # test: from simplex with index
        idx_s = bs.IndexedSimplex(s, index=12)
        assert idx_s is not None
        assert idx_s == s
        assert idx_s.data == s.data
        assert idx_s.weight == s.data
        assert idx_s.dim == s.dimension()
        assert idx_s.index == 12

    def test__from_simplex_with_data(self):
        ss = ds.Simplex((1, 2, 3), 1.3)
        assert ss is not None

        # test: from weighted simplex
        idx_ss = bs.IndexedSimplex(ss)
        assert idx_ss is not None
        assert idx_ss == ss
        assert idx_ss.weight == ss.data
        assert idx_ss.dim == ss.dimension()

        # test: from simplex with data
        idx_ss = bs.IndexedSimplex(ss, data=23)
        assert idx_ss is not None
        assert idx_ss == ss
        assert idx_ss.weight == 23
        assert idx_ss.dim == ss.dimension()

        # test: from simplex with weight
        idx_ss = bs.IndexedSimplex(ss, weight=23)
        assert idx_ss is not None
        assert idx_ss == ss
        assert idx_ss.weight == 23
        assert idx_ss.dim == ss.dimension()

        pass

    def test__from_zero_dim_simplex(self):
        u = ds.Simplex((12, ))
        assert u is not None

        # test: from 0-simplex
        idx_u = bs.IndexedSimplex(u)
        assert idx_u is not None
        assert idx_u == u
        assert idx_u.dim == 0

        # test: from 1-tuple
        idx_u = bs.IndexedSimplex((12, ))
        assert idx_u is not None
        assert idx_u == u
        assert idx_u.dim == 0

        # test: from 1-tuple with data
        idx_u = bs.IndexedSimplex((12, ), data=34)
        assert idx_u is not None
        assert idx_u == u
        assert idx_u.dim == 0
        assert idx_u.weight == 34
        pass

    def test__from_tuple(self):
        s = ds.Simplex((1, 2, 3))
        ss = ds.Simplex((1, 2, 3), 1.3)
        u = ds.Simplex((12,))

        # test: from tuple
        idx_s = bs.IndexedSimplex((1, 2, 3))
        assert idx_s is not None
        assert idx_s == s
        assert idx_s.weight == s.data
        assert idx_s.dim == s.dimension()
        assert idx_s.index is None

        # test: from tuple with data
        idx_ss = bs.IndexedSimplex((1, 2, 3), data=ss.data)
        assert idx_ss is not None
        assert idx_ss == ss
        assert idx_ss.weight == ss.data
        assert idx_ss.dim == ss.dimension()

        # test: from *args
        idx_s = bs.IndexedSimplex(1, 2, 3)
        assert idx_s is not None
        assert idx_s == s
        assert idx_s.weight == s.data
        assert idx_s.dim == s.dimension()
        assert idx_s.index is None

        # test: from *args with data
        idx_ss = bs.IndexedSimplex(1, 2, 3, data=ss.data)
        assert idx_ss is not None
        assert idx_ss == ss
        assert idx_ss.weight == ss.data
        assert idx_ss.dim == ss.dimension()
        pass

    def test__to_string(self):
        s = bs.IndexedSimplex((1, 2, 3), weight=23, index=45)

        bs.IndexedSimplex.VERBOSE_MODE = False
        print(f"\n{s}")
        assert str(s) == "<1,2,3>"

        bs.IndexedSimplex.VERBOSE_MODE = True
        print(f"\n{s}")
        assert str(s) == "<1,2,3> [23.00:45]"

        bs.IndexedSimplex.PRINT_DECIMALS = 0
        print(f"\n{s}")
        assert str(s) == "<1,2,3> [23:45]"

        pass

    pass


class TestPersistenceInterval:
    def test__from_birth_death(self):
        b, d = 3, 10
        bb, dd = 5, 8
        dpoint = ds.DiagramPoint(b, d)

        # test: from birth and death
        pi = bs.PersistenceInterval(b, d)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.lifetime == d - b
        assert pi.finite
        assert pi.data == 0
        assert pi.birth_idx == 0
        assert pi.death_idx is None
        assert pi == dpoint

        # test: from birth, death, and data
        pi = bs.PersistenceInterval(b, d, data=bb)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx is None
        assert pi == dpoint

        # test: from birth, death, birth_idx, and death_idx
        pi = bs.PersistenceInterval(b, d, birth_idx=bb, death_idx=dd)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == dd
        assert pi == dpoint

        # test: birth_idx > data
        pi = bs.PersistenceInterval(b, d, birth_idx=bb, data=dd)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx is None
        assert pi == dpoint

    def test__from_DiagramPoint(self):
        b, d = 3, 10
        bb, dd = 5, 8
        dpoint = ds.DiagramPoint(b, d)
        assert dpoint is not None
        assert dpoint.data == 0
        dpoint.data = bb

        # test: from DiagramPoint
        pi = bs.PersistenceInterval(dpoint)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx is None
        assert pi == dpoint

        # test: from DiagramPoint and data
        pi = bs.PersistenceInterval(dpoint, data=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb            # expects: dpoint.data
        assert pi.birth_idx == bb       # expects: dpoint.data
        assert pi.death_idx is None
        assert pi == dpoint

        # test: from DiagramPoint and birth_idx
        pi = bs.PersistenceInterval(dpoint, birth_idx=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb            # expects: dpoint.data
        assert pi.birth_idx == bb       # expects: dpoint.data
        assert pi.death_idx is None
        assert pi == dpoint

        # test: from DiagramPoint and death_idx
        pi = bs.PersistenceInterval(dpoint, death_idx=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == 7        # expects: 7
        assert pi == dpoint

        pass

    def test__from_PersistenceInterval(self):
        b, d = 3, 10
        bb, dd = 5, 8
        pinterval = bs.PersistenceInterval(b, d, birth_idx=bb, death_idx=dd)

        # test: from PersistenceInterval
        pi = bs.PersistenceInterval(pinterval)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == dd
        assert pi == pinterval

        # test: from PersistenceInterval and data
        pi = bs.PersistenceInterval(pinterval, data=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb            # expects: pinterval.birth_idx
        assert pi.birth_idx == bb       # expects: pinterval.birth_idx
        assert pi.death_idx == dd
        assert pi == pinterval

        # test: from PersistenceInterval and birth_idx
        pi = bs.PersistenceInterval(pinterval, birth_idx=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb            # expects: pinterval.birth_idx
        assert pi.birth_idx == bb       # expects: pinterval.birth_idx
        assert pi.death_idx == dd
        assert pi == pinterval

        # test: from PersistenceInterval and death_idx
        pi = bs.PersistenceInterval(pinterval, death_idx=7)
        assert pi is not None
        assert pi.birth == b
        assert pi.death == d
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == dd       # expects: pinterval.death_idx
        assert pi == pinterval

        pass

    def test__from_infinity(self):
        b, d = 3, 10
        bb, dd = 5, 8

        # test: infinite death
        pi = bs.PersistenceInterval(b, np.inf, birth_idx=bb, death_idx=dd)
        assert pi is not None
        assert pi.birth == b
        assert np.isinf(pi.death)
        assert np.isinf(pi.lifetime)
        assert pi.infinite
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == dd
        assert pi == ds.DiagramPoint(b, np.inf)

        # test: infinite birth
        pi = bs.PersistenceInterval(-np.inf, d, birth_idx=bb, death_idx=dd)
        assert pi is not None
        assert np.isneginf(pi.birth)
        assert pi.death == d
        assert np.isinf(pi.lifetime)
        assert pi.infinite
        assert pi.data == bb
        assert pi.birth_idx == bb
        assert pi.death_idx == dd
        assert pi == ds.DiagramPoint(-np.inf, d)

        pass

    def test__from_exception(self):
        b, d = 3, 10
        bb, dd = 5, 8

        # test: birth > death : finite
        with pytest.raises(ValueError) as error_info:
            pi = bs.PersistenceInterval(d, b, birth_idx=bb, death_idx=dd)
        assert error_info is not None
        assert error_info.value.args[1].birth > error_info.value.args[1].death

        # test: birth > death : infinite birth
        with pytest.raises(ValueError) as error_info:
            pi = bs.PersistenceInterval(np.inf, d, birth_idx=bb, death_idx=dd)
        assert error_info is not None
        assert error_info.value.args[1].birth > error_info.value.args[1].death

        # test: birth > death : infinite death
        with pytest.raises(ValueError) as error_info:
            pi = bs.PersistenceInterval(b, -np.inf, birth_idx=bb, death_idx=dd)
        assert error_info is not None
        assert error_info.value.args[1].birth > error_info.value.args[1].death

        pass

    def test__equals(self):
        # DIAGRAM POINTS:
        dp_1 = ds.DiagramPoint(0, 10)
        dp_1.data = 5
        dp_2 = ds.DiagramPoint(0, 10)
        dp_2.data = 7
        dp_3 = ds.DiagramPoint(3, 10)
        dp_3.data = 5
        print(f"\ndp_1.# = {hash(dp_1)}, dp_2.# = {hash(dp_2)}, dp_3.# = {hash(dp_3)}\n")

        # dp == dp
        assert dp_1 == dp_2
        assert dp_1 != dp_3

        # PERSISTENT INTERVALS:
        pi_1 = bs.PersistenceInterval(
            birth=0, death=10,
            birth_idx=1, death_idx=11, data=5,
        )   # equals: dp_1, dp_2, pi_2;     # not_eq: dp_3, pi_3, pi_4

        pi_2 = bs.PersistenceInterval(
            birth=0, death=10,
            birth_idx=1, death_idx=11, data=7,
        )   # equals: dp_1, dp_2, pi_1;     # not_eq: dp_3, pi_3, pi_4

        pi_3 = bs.PersistenceInterval(
            birth=3, death=10,
            birth_idx=1, death_idx=11, data=5,
        )   # equals: dp_3;                 # not_eq: dp_1, dp_2, pi_1, pi_2, pi_4
        pi_4 = bs.PersistenceInterval(
            birth=0, death=10,
            birth_idx=2, death_idx=11,
            data=5,
        )   # equals: dp_1, dp_2            # not_eq: dp_3, pi_1, pi_2, pi_3

        print(f"\npi_1.# = {hash(pi_1)}, pi_2.# = {hash(pi_2)}, "
              f"pi_3.# = {hash(pi_3)}, pi_4.# = {hash(pi_4)}\n")

        # pi == pi
        assert pi_1 == pi_2
        assert pi_1 != pi_3
        assert pi_1 != pi_4
        assert pi_3 != pi_4
        assert hash(pi_1) == hash(pi_2)
        assert hash(pi_1) != hash(pi_3)
        assert hash(pi_1) != hash(pi_4)
        assert hash(pi_3) != hash(pi_4)

        # pi == dp
        assert pi_1 == dp_1 and dp_1 == pi_1
        assert pi_1 == dp_2 and dp_1 == pi_1
        assert pi_1 != dp_3 and dp_3 != pi_1

        assert pi_3 != dp_1 and dp_1 != pi_3
        assert pi_3 != dp_2 and dp_1 != pi_3
        assert pi_3 == dp_3 and dp_3 == pi_3

        assert pi_4 == dp_1 and dp_1 == pi_4
        assert pi_4 == dp_2 and dp_2 == pi_4
        assert pi_4 != dp_3 and dp_3 != pi_4

        assert hash(pi_1) != hash(dp_1)
        assert hash(pi_1) != hash(dp_2)
        assert hash(pi_1) != hash(dp_3)

        assert hash(pi_3) != hash(dp_1)
        assert hash(pi_3) != hash(dp_2)
        assert hash(pi_3) != hash(dp_3)

        assert hash(pi_4) != hash(dp_1)
        assert hash(pi_4) != hash(dp_2)
        assert hash(pi_4) != hash(dp_3)

        pass

    pass


class TestSlicingFiltration:

    def test__by_dims(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for s in sphere3:
            s.data = s.dimension()
        print(f"sphere3= {list(sphere3)}")
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"filtration= {list(filtration)}")

        dims = [1, 2]
        sf = bs.SlicingFiltration(filtration).by_dims(dims)
        assert list(sf) == [s for s in filtration if s.dimension() in dims]
        assert [sf.index(s) for s in sf] == [filtration.index(s) for s in sf]

        print("\n-----------------")
        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        pass

    def test__by_dims_split(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        filtration = ds.Filtration(sphere3)
        filtration.sort()

        dims = [1, 2]
        sfs = bs.SlicingFiltration(filtration).by_dims(dims, split=True)
        print(sfs)
        assert isinstance(sfs, (tuple, list))
        assert len(sfs) == len(dims), "length == dims :: split by dims"

        assert list(sfs[0]) == [s for s in filtration if s.dimension() == dims[0]]
        assert list(sfs[1]) == [s for s in filtration if s.dimension() == dims[1]]

        assert [sfs[0].index(s) for s in sfs[0]] == [filtration.index(s) for s in sfs[0]]
        assert [sfs[1].index(s) for s in sfs[1]] == [filtration.index(s) for s in sfs[1]]

        print("\nDIM=1")
        for spx in sfs[0]:
            print(f"{str(spx):<11} :: {sfs[0].index(spx):>2}")

        print("\nDIM=2")
        for spx in sfs[1]:
            print(f"{str(spx):<11} :: {sfs[1].index(spx):>2}")

        pass

    def test__by_dims_drop_index(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        filtration = ds.Filtration(sphere3)
        filtration.sort()

        dims = [1, 2]
        sf = bs.SlicingFiltration(filtration).by_dims(dims, drop_index=True)
        assert list(sf) == [s for s in filtration if s.dimension() in dims]
        assert [sf.index(s) for s in sf] == list(range(len(sf)))
        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")
        pass

    def test__by_index_interval(self):
        print()
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for s in sphere3:
            s.data = s.dimension()
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"sphere3 filtration:\n {list(filtration)}\n\n")

        #############################################################
        # first test
        #############################################################
        b, d = interval = [10, 20]
        print(f"\nfirst test: [{b}, {d}]")
        sf = bs.SlicingFiltration(filtration).by_index_interval(*interval)
        assert len(sf) == d - b + 1, "length"
        assert list(sf) == list(filtration)[b:(d + 1)], \
            "s_filtration's content : list"
        assert [sf[i] for i in range(b, d + 1)] == list(filtration)[b:(d + 1)], \
            "s_filtration's content : __getitem__"
        assert [sf.index(s) for s in sf] == list(range(b, d + 1)), \
            "indexes of simplices in s_filtration : Filtration.index()"

        assert (sf.interval.birth, sf.interval.birth_idx) == (filtration[b].dimension(), b), \
            "filtration interval: birth"
        assert (sf.interval.death, sf.interval.death_idx) == (filtration[d].dimension(), d), \
            "filtration interval: death"

        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        #############################################################
        # second test
        #############################################################
        b, d = interval = [13, 17]
        print(f"\nsecond test: [{b}, {d}]")
        sf = sf.by_index_interval(*interval)
        assert len(sf) == d - b + 1, "s_filtration length"
        assert \
            list(sf) == list(filtration)[b:(d + 1)], \
            "s_filtration's content : list"
        assert \
            [sf[i] for i in range(b, d + 1)] == list(filtration)[b:(d + 1)], \
            "s_filtration's content : __getitem__"
        assert \
            [sf.index(s) for s in sf] == list(range(b, d + 1)), \
            "indexes of simplices in s_filtration"

        assert \
            (sf.interval.birth, sf.interval.birth_idx) == (filtration[b].dimension(), b), \
            "filtration interval: birth"
        assert \
            (sf.interval.death, sf.interval.death_idx) ==(filtration[d].dimension(), d), \
            "filtration interval: death"

        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")
        pass

    def test__by_time_interval(self):
        m, M = 5, 63
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for i, s in enumerate(sphere3):
            s.data = 2 * i + m
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"sphere3 filtration: |F| = {len(filtration)}"
              f"\n {list(filtration)}\n\n")

        #############################################################
        # first test
        #############################################################
        b, d = interval = [10, 21]
        print(f"\ntime interval test: [{b}, {d}]")

        sf = bs.SlicingFiltration(filtration).by_time_interval(*interval)
        assert len(sf) == 6, "length"
        assert \
            list(sf) ==[s for s in filtration if b <= s.data <= d], \
            "s_filtration's content : list"

        assert \
            [sf.index(s) for s in sf] == list(range(3, 9)), \
            "indexes of simplices in s_filtration"

        assert sf.interval.birth >= b, "filtration interval: birth"
        assert sf.interval.death <= d, "filtration interval: death"

        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")
        pass

    def test__pop_1(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for s in sphere3:
            s.data = s.dimension()
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"sphere3 filtration:\n {list(filtration)}\n\n")

        # init test
        b, d = interval = [10, 20]
        l = d - b + 1
        print(f"\ninit: [{b}, {d}]")
        sf = bs.SlicingFiltration(filtration).by_index_interval(*interval)
        assert len(sf) == l
        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        # pop test
        n = 1
        print(f"\npop test: n= {n}")
        popped = []
        sf = sf.pop(n, popped=popped)
        print(f"### popped={popped}")
        assert len(sf) == l - n, "filtration length"
        assert len(popped) == n
        assert list(sf) == list(filtration)[b:(d + 1 - n)], \
            "s_filtration content : list"

        assert [sf[i] for i in range(b, d + 1 - n)] == list(filtration)[b:(d + 1 - n)], \
            "s_filtration's content : __getitem__"

        assert [sf.index(s) for s in sf] == list(range(b, d + 1 - n)), \
            "indexes of simplices in s_filtration"

        assert (sf.interval.birth, sf.interval.birth_idx) == (filtration[b].dimension(), b), \
            "s_filtration interval: birth"

        assert (sf.interval.death, sf.interval.death_idx) == (filtration[d - n].dimension(), d - n), \
            "s_filtration interval: death"

        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        pass

    def test__pop_2(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for s in sphere3:
            s.data = s.dimension()
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"sphere3 filtration:\n {list(filtration)}\n\n")

        # init test
        b, d = interval = [10, 20]
        l = d - b + 1
        print(f"\ninit: [{b}, {d}]")
        sf = bs.SlicingFiltration(filtration).by_index_interval(*interval)
        assert len(sf) == l
        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        # pop test
        n = 2
        print(f"\npop test: n= {n}")
        popped = []
        sf = sf.pop(n, popped=popped)
        print(f"### popped={popped}")
        assert len(sf) == l - n, "filtration length"
        assert len(popped) == n
        assert list(sf) == list(filtration)[b:(d + 1 - n)], \
            "s_filtration content : list"
        assert [sf[i] for i in range(b, d + 1 - n)] ==list(filtration)[b:(d + 1 - n)], \
            "s_filtration's content : __getitem__"
        assert [sf.index(s) for s in sf] == list(range(b, d + 1 - n)), \
            "indexes of simplices in s_filtration"

        assert \
            (sf.interval.birth, sf.interval.birth_idx) == (filtration[b].dimension(), b), \
            "s_filtration interval: birth"
        assert \
            (sf.interval.death, sf.interval.death_idx) == (filtration[d - n].dimension(), d - n), \
            "s_filtration interval: death"

        for spx in sf:
            print(f"{str(spx):<11} :: {sf.index(spx):>2}")

        pass

    def test__to_list(self):
        simplex4 = ds.Simplex([0, 1, 2, 3, 4], 10.0)
        sphere3 = ds.closure([simplex4], 3)
        for s in sphere3:
            s.data = s.dimension()
        filtration = ds.Filtration(sphere3)
        filtration.sort()
        print(f"sphere3 filtration:\n {list(filtration)}\n\n")

        #############################################################
        # first test
        #############################################################
        b, d = interval = [10, 20]
        print(f"\nfirst test: [{b}, {d}]")
        sf = bs.SlicingFiltration(filtration).by_index_interval(*interval)
        assert len(sf) == d - b + 1, "length"

        sf_list = sf.to_list()
        assert [s.index for s in sf_list] == list(range(b, d + 1))

        pass


    pass






