import pytest
from collections import Counter

import numpy as np

from cyclonysus import Cycler

class TestCycler():
    def test_from_simplices(self):
        sim = [
            ([0], 0.3),
            ([1], 0.5),
            ([2], 0.6),
            ([0, 1], 0.7),
            ([1, 2], 0.9),
            ([0, 2], 0.99)
        ]

        cycler = Cycler()
        cycler.from_simplices(sim)
        longest = cycler.longest_intervals(1)[0]
        cycle = cycler.get_cycle(longest)

        vertices = np.array(cycle).flatten()
        assert 0 in vertices and 1 in vertices and 2 in vertices
        pass

    def test_is_a_cycle(self):
        cycler = Cycler()
        data = np.random.random((100, 3))

        cycler.fit(data)
        longest = cycler.longest_intervals(1)[0]
        cycle = cycler.get_cycle(longest)

        vertices = np.array(cycle).flatten()
        counts = Counter(vertices)
        
        assert all([c == 2 for c in counts.values()])
        pass

    def test_longest_intervals(self):
        cycler = Cycler()
        data = np.random.random((100, 3))

        cycler.fit(data)

        longest = cycler.longest_intervals(1)
        assert len(longest) == 1

        longest = cycler.longest_intervals(4)
        assert len(longest) == 4
        pass

