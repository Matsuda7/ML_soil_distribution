"""Select model-specific stage hooks without coupling the common pipeline to them."""
import importlib
from module.configuration import load_model_settings

BACKENDS = {'xgboost': 'module.backends.xgboost', 'tabpfn': 'module.backends.tabpfn'}


def selected_model():
    name = load_model_settings().get('model', 'xgboost')
    if name == 'tabpfn' and name not in BACKENDS:
        raise ValueError('model=tabpfn is reserved for a future backend and is not implemented yet; no fallback to xgboost')
    if not isinstance(name, str) or name not in BACKENDS:
        raise ValueError(f'Model {name!r} is not implemented; available models: {", ".join(BACKENDS)}')
    return name


def get_backend(name=None):
    return importlib.import_module(BACKENDS[name or selected_model()])
