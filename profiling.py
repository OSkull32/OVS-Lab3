from . import networks
from .simulation import _random_input, run_network


def profile(names=None, pe_counts=(3, 4, 5, 6, 7), seed=42):
    """
    Собрать метрики по сетям и числу ядер.
    """
    if names is None:
        names = list(networks.STRUCTURES.keys())
    results = {
        "networks": list(names),
        "pe_counts": list(pe_counts),
        "time": {},
        "memory": {},
        "load": {},
        "total_macs": {},
        "params": {},
    }
    for name in names:
        net = networks.build_network(name, seed=seed)
        sizes = net["sizes"]
        inp = _random_input(sizes[0])
        n_hidden = len(sizes) - 2
        results["params"][name] = networks.parameter_count(sizes)
        results["total_macs"][name] = sum(
            sizes[k] * sizes[k + 1] for k in range(len(sizes) - 1)
        )
        results["time"][name] = {}
        results["memory"][name] = {}
        results["load"][name] = {}
        for pe in pe_counts:
            res = run_network(net, pe, inp)
            results["time"][name][pe] = res["time"]
            results["load"][name][pe] = res["pe_macs"]
            local_total = sum(res["local_words"])  # замер локальной памяти из модели
            config = n_hidden + 5
            lut = networks.LUT_WORDS
            results["memory"][name][pe] = {
                "shared": res["shared_words"],  # замер shared-памяти из модели
                "local_total": local_total,
                "config": config,
                "lut": lut,
                "total": res["shared_words"] + local_total + config + lut,
            }
    return results


def _imbalance(macs):
    """Дисбаланс загрузки ядер: (max - min) / max."""
    return (max(macs) - min(macs)) / max(macs) if max(macs) else 0.0


def print_tables(data, ref_pe=4):
    names = data["networks"]
    pes = data["pe_counts"]

    print("TIME (messages = model time):")
    header = "net  " + "".join(f"{'PE=' + str(p):>8}" for p in pes)
    print(header)
    for n in names:
        row = f"{n:5s}" + "".join(f"{data['time'][n][p]:>8d}" for p in pes)
        print(row)

    pe = ref_pe if ref_pe in pes else pes[0]
    print()
    print(f"MEMORY (words, measured from model, PE={pe}):")
    print("net  shared  local_total  config  lut  total")
    for n in names:
        m = data["memory"][n][pe]
        print(
            f"{n:5s}  {m['shared']:6d}  {m['local_total']:11d}  {m['config']:6d}  "
            f"{m['lut']:4d}  {m['total']:6d}"
        )

    print()
    print(f"LOAD (MAC per PE, PE={pe}) and imbalance:")
    for n in names:
        macs = data["load"][n][pe]
        print(f"{n:5s}  macs/PE={macs}  total={sum(macs)}  imbalance={_imbalance(macs):.2%}")


def main():
    print_tables(profile())


if __name__ == "__main__":
    main()
