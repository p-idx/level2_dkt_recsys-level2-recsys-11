import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


def get_model(args):

    model_name = args.model

    if model_name == "XGB":
        model = xgb.XGBClassifier(**args, n_estimators=100, random_state=args.SEED)

    if model_name == 'LGB':
        param = {'objective': 'binary',
                    'metric': 'auc',
                    'boosting': 'dart',
                    'learning_rate': 0.03,
                    'max_depth': 64,
                    'num_leaves': 63
                    }
        model = lgb.LGBMClassifier(**param, n_estimators=100) #need seed

    if model_name == 'CATB':
        model = ctb.CatBoostClassifier(
                                    eval_metric='AUC',
                                    iterations=args.n_epochs,
                                    depth=args.depth,
                                    learning_rate=args.lr,
                                    verbose=args.verbose,
                                    loss_function='Logloss', #사용자 지정 로스도 가능한 모양
                                    od_type='IncToDec',
                                    od_pval=args.od_pval,
                                    od_wait=args.od_wait,
                                    )
    return model