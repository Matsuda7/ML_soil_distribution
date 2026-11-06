"""TabPFN stage hooks. Stages 02, 03 and 04 support effective cohesion."""
NAME = 'tabpfn'


def prepare(targets):
    if not targets or set(targets) != {'c'}:
        raise NotImplementedError('TabPFN stage 02 currently supports only --targets c')
    from module.tabpfn_preparation import prepare_cohesion
    return {'c': prepare_cohesion()}


def train(target,gamma_kind=None,n_distribution_run=None,prepared_data=None):
    if target!='c':raise NotImplementedError('TabPFN stage 03 currently supports only c')
    if n_distribution_run is not None:
        raise NotImplementedError('TabPFN c stage 03 uses the N distribution frozen by stage 02; prepare again after the required phi run')
    from module.tabpfn_training import train_cohesion
    return train_cohesion(prepared_data=prepared_data)


def predict(target,run,n_distribution_run=None):
    if target!='c':raise NotImplementedError('TabPFN stage 04 currently supports only c')
    from module.tabpfn_prediction import predict_cohesion
    return predict_cohesion(run,n_distribution_run=n_distribution_run)
