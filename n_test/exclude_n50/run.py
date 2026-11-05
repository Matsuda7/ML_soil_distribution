"""Compatibility entry point for the N=50 exclusion experiment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("module.experiments.exclude_n50", run_name="__main__")
