from .. import networks
from ..profiling import profile
from ..simulation import expected_messages

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = profile()
    return _DATA


def test_time_equals_expected():
    data = _data()
    for n in data["networks"]:
        sizes = networks.STRUCTURES[n]
        for p in data["pe_counts"]:
            assert data["time"][n][p] == expected_messages(sizes, p), (n, p)


def test_memory_matches_formula():
    data = _data()
    for n in data["networks"]:
        sizes = networks.STRUCTURES[n]
        for p in data["pe_counts"]:
            formula = networks.memory_words(sizes, p)
            assert data["memory"][n][p]["shared"] == formula["shared"], (n, p)
            assert data["memory"][n][p]["local_total"] == formula["local_total"], (n, p)


def test_load_sums_to_total_macs():
    data = _data()
    for n in data["networks"]:
        for p in data["pe_counts"]:
            assert sum(data["load"][n][p]) == data["total_macs"][n], (n, p)


def test_time_grows_with_pe():
    data = _data()
    for n in data["networks"]:
        for i in range(1, len(data["pe_counts"])):
            p, prev = data["pe_counts"][i], data["pe_counts"][i - 1]
            assert data["time"][n][p] > data["time"][n][prev], (n, p)


def test_shared_memory_independent_of_pe():
    data = _data()
    for n in data["networks"]:
        vals = {data["memory"][n][p]["shared"] for p in data["pe_counts"]}
        assert len(vals) == 1, n


if __name__ == "__main__":
    import sys

    from . import run_module

    sys.exit(0 if run_module(globals()) else 1)
