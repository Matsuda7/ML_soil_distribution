"""Compatibility API; numbered pipeline calls backend stages directly."""
def distribute(target,gamma_kind=None,n_distribution_run=None):
    from module.model_backend import get_backend
    backend=get_backend()
    trained=backend.train(target,gamma_kind=gamma_kind,n_distribution_run=n_distribution_run)
    return backend.predict(target,trained['output'],n_distribution_run=n_distribution_run)
