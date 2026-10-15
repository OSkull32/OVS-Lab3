import random

import simpy

from . import networks
from .model import ActivationCore, BusMatrix, Channel, ProcessingElement, SharedMemory, sigmoid


def build_memory(network):
    """
    Разложить описание сети в SHARED MEMORY и вернуть карту адресов.

    Возвращает базовые адреса весов (w_base), активаций (act_base), входа и результата.
    """
    sizes = network["sizes"]
    weights = network["weights"]
    biases = network["biases"]
    num_layers = len(sizes) - 1
    num_hidden = len(sizes) - 2

    # Заголовок: I, O, N, H1..HN
    sm_data = {0: sizes[0], 1: sizes[-1], 2: num_hidden}
    for i, h in enumerate(sizes[1:-1]):
        sm_data[3 + i] = h

    # Базовые адреса весов каждого слоя (n_out блоков по n_in+1 слов)
    offset = 3 + num_hidden
    w_base = []
    for k in range(num_layers):
        n_in, n_out = sizes[k], sizes[k + 1]
        w_base.append(offset)
        offset += n_out * (n_in + 1)

    # Входные значения
    input_addr = offset
    offset += sizes[0]

    # Активации скрытых слоёв
    act_base = []
    for k in range(num_layers - 1):
        act_base.append(offset)
        offset += sizes[k + 1]

    result_addr = offset

    # Заполнение весов и смещений: для нейрона i слоя k - n_in весов, затем смещение
    for k in range(num_layers):
        n_in, n_out = sizes[k], sizes[k + 1]
        for i in range(n_out):
            base = w_base[k] + i * (n_in + 1)
            for j in range(n_in):
                sm_data[base + j] = weights[k][j][i]
            sm_data[base + n_in] = biases[k][i]

    return {
        "sizes": sizes,
        "num_layers": num_layers,
        "w_base": w_base,
        "act_base": act_base,
        "input_addr": input_addr,
        "result_addr": result_addr,
        "sm_data": sm_data,
    }


def reference_forward(network, inputs):
    """Эталонный проход - для проверки модели."""
    sizes = network["sizes"]
    weights = network["weights"]
    biases = network["biases"]
    a = list(inputs)
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        new_a = []
        for i in range(n_out):
            s = biases[k][i]
            for j in range(n_in):
                s += weights[k][j][i] * a[j]
            new_a.append(sigmoid(s))
        a = new_a
    return a


def expected_messages(sizes, num_pe):
    """
    Аналитическая формула модельного времени.

    T = I + O + Σ_k [ P · n_in_k + n_out_k · (n_in_k + 3) ],
    где на нейрон: n_in+1 слово весов + 1 сообщение (сумма) + 1 сообщение (активация).
    """
    total = sizes[0] + sizes[-1]
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        total += num_pe * n_in + n_out * (n_in + 3)
    return total


def run_network(network, num_pe, input_values):
    """
    Прогнать одну сеть на `num_pe` ядрах и вернуть выход + метрики.
    """
    layout = build_memory(network)
    sizes = layout["sizes"]
    num_layers = layout["num_layers"]

    env = simpy.Environment()
    bus = BusMatrix(env)
    sm = SharedMemory(env, bus)
    act_channel = Channel(env, bus)
    # Загрузка описания сети в память (конфигурирование) - не входит в расчёт времени
    for addr, v in layout["sm_data"].items():
        sm.mem[addr] = v

    pes = [ProcessingElement(i, env, bus, sm, act_channel) for i in range(num_pe)]
    act = ActivationCore(env, bus, sm, act_channel)

    start = simpy.Event(env)   # IO сообщает о готовности входа
    done = simpy.Event(env)    # CONTROL UNIT сообщает о завершении
    output = {}

    def io_ctrl():
        """IO CONTROLLER: пишет вход, затем читает результат и считает argmax."""
        for i, v in enumerate(input_values):
            yield from sm.write(layout["input_addr"] + i, v)
        start.succeed()
        yield done
        outs = []
        for i in range(sizes[-1]):
            outs.append((yield from sm.read(layout["result_addr"] + i)))
        output["values"] = outs
        output["class"] = max(range(len(outs)), key=lambda i: outs[i])

    def control():
        """CONTROL UNIT: по слоям транслирует входы и распределяет нейроны по ядрам."""
        yield start
        for k in range(num_layers):
            n_in, n_out = sizes[k], sizes[k + 1]
            # Входы слоя: для первого слоя - изображение, иначе активации прошлого слоя
            if k == 0:
                in_addrs = list(range(layout["input_addr"], layout["input_addr"] + n_in))
            else:
                in_addrs = list(range(layout["act_base"][k - 1], layout["act_base"][k - 1] + n_in))
            # Трансляция входов на все ядра (P · n_in сообщений)
            for pe in pes:
                yield from pe.load_inputs(in_addrs)
            # Распределение нейронов: round-robin по свободным ядрам
            for i in range(n_out):
                pe = pes[i % num_pe]
                w_addrs = [layout["w_base"][k] + i * (n_in + 1) + j for j in range(n_in + 1)]
                res_addr = (
                    layout["act_base"][k] + i
                    if k < num_layers - 1
                    else layout["result_addr"] + i
                )
                yield from pe.load_weights(w_addrs)
                s = yield from pe.compute_neuron()
                yield from act_channel.recv()
                yield from act.write_activation(res_addr, s)
        done.succeed()

    env.process(io_ctrl())
    env.process(control())
    env.run()

    return {
        "output": output.get("values"),
        "class": output.get("class"),
        "time": env.now,
        "messages": bus.word_count,
        "pe_neurons": [pe.neurons for pe in pes],
        "pe_macs": [pe.macs for pe in pes],
        "shared_words": len(sm.mem),
        "local_words": [pe.max_local for pe in pes],
        "sm_reads": sm.reads,
        "sm_writes": sm.writes,
        "activations": act.activations,
    }


def _random_input(size, seed=7):
    """Случайное бинарное входное изображение (0/1) с фиксированным seed."""
    rng = random.Random(seed)
    return [float(rng.randint(0, 1)) for _ in range(size)]


def main():
    """Демонстрация: сеть N1 при разном числе ядер."""
    net = networks.build_network("N1", seed=42)
    sizes = net["sizes"]
    inp = _random_input(sizes[0])
    ref = reference_forward(net, inp)
    print("reference:", [round(v, 4) for v in ref])
    for num_pe in [3, 4, 7]:
        res = run_network(net, num_pe, inp)
        print(
            f"PE={num_pe}: output={[round(v, 4) for v in res['output']]} "
            f"class={res['class']} time={res['time']} "
            f"neurons/PE={res['pe_neurons']} shared={res['shared_words']}"
        )


if __name__ == "__main__":
    main()
