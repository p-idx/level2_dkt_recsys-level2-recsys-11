import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


def get_model(cfg):

    model_name = cfg.model_name

    if model_name == "XGB":
        model = XGBClassifier(**cfg, n_estimators=100, random_state=args.SEED)

    if model_name == 'LGBM':
        model = LGBMClassifier(**cfg.model_param, n_estimators=100, random_state=args.SEED)

    if model_name == 'CATB':
        model = CatBoostClassifier(**cfg.model_param)

    return model