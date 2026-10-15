import importlib
import sys

MODULES = [
    "lab3.tests.test_model",
    "lab3.tests.test_simulation",
    "lab3.tests.test_profiling",
]


def run():
    passed = 0
    failed = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        print(f"== {mod_name} ==")
        name_of = mod.__name__
        for key in sorted(dir(mod)):
            if not key.startswith("test_"):
                continue
            fn = getattr(mod, key)
            if getattr(fn, "__module__", None) != name_of:
                continue
            try:
                fn()
                print(f"  PASS {key}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL {key}: {exc!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
