import math

import simpy


def sigmoid(x):
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class BusMatrix:
    """
    Коммутационная среда: все пересылки сериализуются (ёмкость 1).

    `transfer(words)` занимает шину на `words` единиц времени (1 слово = 1 сообщение);
    счётчик `word_count` накапливает переданные слова и равен модельному времени.
    """

    def __init__(self, env):
        self.env = env
        self.resource = simpy.Resource(env, capacity=1)
        self.word_count = 0

    def transfer(self, words=1):
        with self.resource.request() as req:
            yield req
            yield self.env.timeout(words)
            self.word_count += words


class SharedMemory:
    """
    Общая память: описание сети, входы, промежуточные активации, результат.
    Каждое обращение идёт через шину и стоит 1 единицу времени.
    """

    def __init__(self, env, bus):
        self.env = env
        self.bus = bus
        self.mem = {}
        self.reads = 0
        self.writes = 0

    def read(self, addr):
        yield from self.bus.transfer(1)
        self.reads += 1
        return self.mem.get(addr, 0)

    def write(self, addr, value):
        yield from self.bus.transfer(1)
        self.writes += 1
        self.mem[addr] = value


class Channel:
    """
    Канал точка-точка между блоками.
    Отправка сообщения занимает шину (1 единица времени), приём - без времени.
    """

    def __init__(self, env, bus):
        self.env = env
        self.bus = bus
        self.store = simpy.Store(env)
        self.messages = 0

    def send(self, value):
        yield from self.bus.transfer(1)
        self.store.put(value)
        self.messages += 1

    def recv(self):
        value = yield self.store.get()
        return value


class ProcessingElement:
    """
    Вычислительное ядро PE(MUL): считает взвешенную сумму одного нейрона.

    Локальная память ядра - две очереди: входов (`inputs`) и весов (`weights`).
    Загрузка очередей идёт через шину (сообщения), само вычисление - мгновенно.
    """

    def __init__(self, pe_id, env, bus, sm, act_channel):
        self.pe_id = pe_id
        self.env = env
        self.bus = bus
        self.sm = sm
        self.act_channel = act_channel
        self.inputs = []      # INPUT QUEUE (локальная память)
        self.weights = []     # WEIGHT QUEUE (локальная память), последний элемент - смещение
        self.max_local = 0    # пиковый объём локальной памяти (слов)
        self.macs = 0         # число выполненных умножений (MAC)
        self.neurons = 0      # число обработанных нейронов

    @staticmethod
    def weighted_sum(weights, inputs):
        return sum(w * x for w, x in zip(weights, inputs))

    def load_inputs(self, in_addrs):
        self.inputs = []
        for a in in_addrs:
            v = yield from self.sm.read(a)
            self.inputs.append(v)

    def load_weights(self, w_addrs):
        self.weights = []
        for a in w_addrs:
            v = yield from self.sm.read(a)
            self.weights.append(v)
        self.max_local = max(self.max_local, len(self.inputs) + len(self.weights))

    def compute_neuron(self):
        bias = self.weights[-1]
        s = bias + self.weighted_sum(self.weights[:-1], self.inputs)
        self.macs += len(self.inputs)
        self.neurons += 1
        yield from self.act_channel.send(s)
        return s


class ActivationCore:
    """
    Ядро активации: применяет сигмоиду и пишет активацию в память.
    """

    def __init__(self, env, bus, sm, act_channel):
        self.env = env
        self.bus = bus
        self.sm = sm
        self.act_channel = act_channel
        self.activations = 0

    @staticmethod
    def activate(s):
        return sigmoid(s)

    def write_activation(self, res_addr, s):
        y = self.activate(s)
        yield from self.sm.write(res_addr, y)
        self.activations += 1
        return y
