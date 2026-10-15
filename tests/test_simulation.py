from .. import networks
from ..simulation import _random_input, expected_messages, reference_forward, run_network

PE_COUNTS = (3, 4, 5, 6, 7)


def test_output_matches_reference():
    for name in networks.STRUCTURES:
        net = networks.build_network(name, seed=42)
        inp = _random_input(net["sizes"][0])
        ref = reference_forward(net, inp)
        for pe in PE_COUNTS:
            res = run_network(net, pe, inp)
            assert all(abs(a - b) < 1e-9 for a, b in zip(res["output"], ref)), (name, pe)


def test_time_equals_message_count():
    for name in networks.STRUCTURES:
        sizes = networks.STRUCTURES[name]
        net = networks.build_network(name, seed=42)
        inp = _random_input(sizes[0])
        for pe in PE_COUNTS:
            res = run_network(net, pe, inp)
            assert res["time"] == expected_messages(sizes, pe), (name, pe)
            assert res["messages"] == res["time"], (name, pe)


def test_output_independent_of_pe_count():
    for name in networks.STRUCTURES:
        net = networks.build_network(name, seed=42)
        inp = _random_input(net["sizes"][0])
        outputs = [run_network(net, pe, inp)["output"] for pe in PE_COUNTS]
        for out in outputs[1:]:
            assert all(abs(a - b) < 1e-9 for a, b in zip(out, outputs[0])), name


if __name__ == "__main__":
    import sys

    from . import run_module

    sys.exit(0 if run_module(globals()) else 1)
