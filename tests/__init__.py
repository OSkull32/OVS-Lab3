def run_module(module_globals):
    name_of = module_globals["__name__"]
    passed = 0
    failed = 0
    for key in sorted(module_globals):
        if not key.startswith("test_"):
            continue
        fn = module_globals[key]
        if not callable(fn) or getattr(fn, "__module__", None) != name_of:
            continue
        try:
            fn()
            print(f"PASS {key}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {key}: {exc!r}")
            failed += 1
    print(f"{passed} passed, {failed} failed")
    return failed == 0
