
def XGB(cfg):
    pass

def LGBM(cfg):
    pass

def CATB(cfg):
    pass

def get_model(model_name: str, cfg: dict):
    if model_name == "XGB":
        model = XGB(cfg)
    if model_name == 'LGBM':
        model = LGBM(cfg)
    if model_name == 'CATB':
        model = CATB(cfg)

    return model