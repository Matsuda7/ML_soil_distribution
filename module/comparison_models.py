"""XGBoost estimator construction and bounded-memory prediction batches."""
import numpy as np
from module.xgb_common import build_regressor


def build_model(name, config, rows, columns):
    if name != 'xgboost':
        raise ValueError('Only xgboost is supported')
    params = dict(config['xgboost'])
    if 'early_stopping_rounds' in params or 'callbacks' in params:
        raise ValueError('Spatial evaluation uses fixed parameters; no outer-fold tuning')
    params['random_state'] = config['random_state']
    return build_regressor(params)


def predict_batches(model, values, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    result = [np.asarray(model.predict(values[start:start + batch_size]), dtype=float)
              for start in range(0, len(values), batch_size)]
    predicted = np.concatenate(result) if result else np.empty(0)
    if not np.isfinite(predicted).all():
        raise ValueError('Non-finite predictions')
    return predicted


def save_model(model, name, directory):
    if name != 'xgboost':
        raise ValueError('Only xgboost is supported')
    model.save_model(directory / 'model.json')
