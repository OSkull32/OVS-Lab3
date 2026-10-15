import simpy

from ..model import ActivationCore, BusMatrix, Channel, ProcessingElement, SharedMemory, sigmoid


def test_bus_serialization():
    env = simpy.Environment()
    bus = BusMatrix(env)

    def worker(n):
        for _ in range(n):
            yield from bus.transfer(1)

    env.process(worker(3))
    env.process(worker(3))
    env.run()
    assert bus.word_count == 6
    assert env.now == 6


def test_shared_memory():
    env = simpy.Environment()
    bus = BusMatrix(env)
    sm = SharedMemory(env, bus)
    result = {}

    def seq():
        yield from sm.write(5, 99)
        result["v"] = yield from sm.read(5)

    env.process(seq())
    env.run()
    assert result["v"] == 99
    assert sm.reads == 1 and sm.writes == 1
    assert env.now == 2


def test_channel():
    env = simpy.Environment()
    bus = BusMatrix(env)
    ch = Channel(env, bus)
    result = {}

    def sender():
        yield from ch.send(7)

    def receiver():
        result["v"] = yield from ch.recv()

    env.process(sender())
    env.process(receiver())
    env.run()
    assert result["v"] == 7
    assert ch.messages == 1
    assert env.now == 1


def test_compute_without_time():
    env = simpy.Environment()

    def compute():
        assert sum(i * i for i in range(1000)) == 332833500
        assert env.now == 0
        if False:
            yield

    env.process(compute())
    env.run()
    assert env.now == 0


def test_sigmoid_matches_reference():
    from neural_network import sigmoid as ref_sigmoid

    for x in [-3.0, -1.0, 0.0, 0.5, 2.0, 5.0]:
        assert abs(sigmoid(x) - ref_sigmoid(x)) < 1e-12, x


def test_pe_and_activation():
    env = simpy.Environment()
    bus = BusMatrix(env)
    sm = SharedMemory(env, bus)
    ch = Channel(env, bus)
    pe = ProcessingElement(0, env, bus, sm, ch)
    act = ActivationCore(env, bus, sm, ch)

    inputs = [0.5, 0.3]
    weights = [0.2, -0.4]
    bias = 0.1
    expected = sigmoid(bias + ProcessingElement.weighted_sum(weights, inputs))

    def orchestrator():
        for i, v in enumerate(inputs):
            yield from sm.write(i, v)
        for i, v in enumerate(weights):
            yield from sm.write(10 + i, v)
        yield from sm.write(13, bias)
        yield from pe.load_inputs([0, 1])
        yield from pe.load_weights([10, 11, 13])
        s = yield from pe.compute_neuron()
        yield from ch.recv()
        yield from act.write_activation(20, s)

    env.process(orchestrator())
    env.run()

    assert abs(sm.mem[20] - expected) < 1e-12
    assert pe.macs == 2
    assert pe.neurons == 1
    assert pe.max_local == 5
    assert act.activations == 1
    assert env.now == 12


if __name__ == "__main__":
    import sys

    from . import run_module

    sys.exit(0 if run_module(globals()) else 1)
