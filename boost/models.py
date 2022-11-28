import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


def get_model(args):

    model_name = args.model

    if model_name == "XGB":
        model = xgb.XGBClassifier(**args, n_estimators=100, random_state=args.SEED)

    if model_name == 'LGBM':
        model = lgb.LGBMClassifier(**args.model_param, n_estimators=100, random_state=args.SEED)

    if model_name == 'CATB':
        model = ctb.CatBoostRegressor(
                                    eval_metric='AUC',
                                    iterations=args.n_epochs,
                                    depth=args.depth,
                                    learning_rate=args.lr,
                                    verbose=50,
                                    loss_function='RMSE', #사용자 지정 로스도 가능한 모양
                                    od_type='IncToDec',
                                    od_pval=10,
                                    od_wait=50,
                                    )
    return model