import argparse

from . import networks
from .simulation import _random_input, reference_forward, run_network


def _cmd_run(args):
    if args.layers:
        # Произвольная структура, например --layers 49,16,16,3
        net = networks.build_from_sizes(networks.parse_sizes(args.layers), seed=args.seed)
    else:
        # Готовая сеть N1..N4
        net = networks.build_network(args.network, seed=args.seed)
    inp = _random_input(net["sizes"][0], seed=args.input_seed)
    res = run_network(net, args.pe, inp)
    ref = reference_forward(net, inp)
    topo = "->".join(str(s) for s in net["sizes"])
    print(f"network={topo}  PE={args.pe}  seed={args.seed}")
    print(f"reference: {[round(v, 4) for v in ref]}")
    print(f"model:     {[round(v, 4) for v in res['output']]}  class={res['class']}")
    print(f"time={res['time']} messages  shared={res['shared_words']} words")
    print(f"neurons/PE={res['pe_neurons']}")
    print(f"macs/PE={res['pe_macs']}")


def _cmd_profile(args):
    from .profiling import print_tables, profile

    names = args.nets.split(",") if args.nets else None
    pes = tuple(int(x) for x in args.pes.split(","))
    data = profile(names=names, pe_counts=pes, seed=args.seed)
    print_tables(data, ref_pe=args.ref_pe)


def _cmd_test(args):
    from .tests import run_all

    raise SystemExit(0 if run_all.run() else 1)


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m lab3", description="ЛР3: модель нейросетевого процессора"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="прогнать одну сеть и сравнить с эталоном")
    r.add_argument("--network", default="N1", choices=list(networks.STRUCTURES))
    r.add_argument("--layers", default=None, help="своя структура, напр. 49,32,3")
    r.add_argument("--pe", type=int, default=4, help="число PE")
    r.add_argument("--seed", type=int, default=42, help="seed весов")
    r.add_argument("--input-seed", type=int, default=7, help="seed входного изображения")
    r.set_defaults(func=_cmd_run)

    pr = sub.add_parser("profile", help="профилирование: таблицы время/память/загрузка")
    pr.add_argument("--nets", default=None, help="сети через запятую, напр. N1,N2,N3")
    pr.add_argument("--pes", default="3,4,5,6,7", help="числа PE через запятую")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--ref-pe", type=int, default=4, help="PE для таблиц памяти/загрузки")
    pr.set_defaults(func=_cmd_profile)

    ts = sub.add_parser("test", help="запустить тесты")
    ts.set_defaults(func=_cmd_test)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
