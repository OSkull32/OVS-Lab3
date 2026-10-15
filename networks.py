import random

# Профилируемые структуры: [вход, скрытые слои..., выход]
STRUCTURES = {
    "N1": [49, 32, 3],
    "N2": [49, 16, 16, 3],
    "N3": [49, 64, 32, 3],
    "N4": [49, 8, 3],
}

# Ожидаемое число параметров (веса + смещения)
EXPECTED_PARAMS = {
    "N1": 1699,
    "N2": 1123,
    "N3": 5379,
    "N4": 427,
}

WORD_BYTES = 2   # один элемент (слово) = 16 бит
LUT_WORDS = 512   # таблица сигмоиды: 256 сегментов × 2 коэффициента (A, B)


def parameter_count(sizes):
    total = 0
    for k in range(len(sizes) - 1):
        total += sizes[k] * sizes[k + 1] + sizes[k + 1]
    return total


def generate_weights(sizes, seed=42):
    """
    Детерминированная генерация весов и смещений в диапазоне [-0.5, 0.5].
    """
    rng = random.Random(seed)
    weights = []
    biases = []
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        w = [[rng.uniform(-0.5, 0.5) for _ in range(n_out)] for _ in range(n_in)]
        b = [rng.uniform(-0.5, 0.5) for _ in range(n_out)]
        weights.append(w)
        biases.append(b)
    return weights, biases


def build_network(name, seed=42):
    sizes = STRUCTURES[name]
    weights, biases = generate_weights(sizes, seed)
    return {"name": name, "sizes": sizes, "weights": weights, "biases": biases}


def build_from_sizes(sizes, seed=42, name="custom"):
    sizes = list(sizes)
    weights, biases = generate_weights(sizes, seed)
    return {"name": name, "sizes": sizes, "weights": weights, "biases": biases}


def parse_sizes(text):
    return [int(x) for x in text.split(",") if x.strip()]


def memory_words(sizes, num_pe):
    """
    Объём памяти в 16-битных словах.
    """
    n_hidden = len(sizes) - 2
    params = parameter_count(sizes)
    hidden_activations = sum(sizes[1:-1])
    shared = (3 + n_hidden) + params + sizes[0] + hidden_activations + sizes[-1]
    max_input = max(sizes[:-1])
    per_pe = 2 * max_input + 1
    local_total = num_pe * per_pe
    config = n_hidden + 5
    return {
        "shared": shared,
        "hidden_activations": hidden_activations,
        "local_per_pe": per_pe,
        "local_total": local_total,
        "config": config,
        "lut": LUT_WORDS,
        "total": shared + local_total + config + LUT_WORDS,
    }


def memory_bytes(sizes, num_pe):
    mem = memory_words(sizes, num_pe)
    return {k: v * WORD_BYTES for k, v in mem.items()}


def main():
    print("name  topology          params  expected  ok")
    for name in ["N1", "N2", "N3", "N4"]:
        sizes = STRUCTURES[name]
        params = parameter_count(sizes)
        ok = "OK" if params == EXPECTED_PARAMS[name] else "FAIL"
        topo = "->".join(str(s) for s in sizes)
        print(f"{name:4s}  {topo:18s} {params:6d}  {EXPECTED_PARAMS[name]:6d}  {ok}")

    print()
    print("memory (words, PE=4):")
    print("name  shared  local_per_pe  local_total  config  lut   total")
    for name in ["N1", "N2", "N3", "N4"]:
        sizes = STRUCTURES[name]
        m = memory_words(sizes, 4)
        print(
            f"{name:4s}  {m['shared']:6d}  {m['local_per_pe']:12d}  "
            f"{m['local_total']:11d}  {m['config']:6d}  {m['lut']:5d}  {m['total']:6d}"
        )

    print()
    print("total memory depends on PE count (N1):")
    for pe in range(3, 8):
        m = memory_words(STRUCTURES["N1"], pe)
        print(f"  PE={pe}: total={m['total']} words ({m['total'] * WORD_BYTES} bytes)")


if __name__ == "__main__":
    main()
