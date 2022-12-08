import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


def get_model(args):

    model_name = args.model

    if model_name == "XGB":
        model = xgb.XGBClassifier(**args, n_estimators=100, random_state=args.SEED)

    if model_name == 'LGB':
        param = {'objective': 'binary',
                'metric': ['auc', 'binary_logloss'],
                'boosting_type': 'goss', # gbdt, dart, rf, goss
                'learning_rate': args.lr,
                'max_depth': args.lgb_depth,
                'num_leaves': args.lgb_leaves,
                'n_estimators': args.n_epochs,
                'min_child_samples': args.lgb_child_samples,
                }
        model = lgb.LGBMClassifier(**param) #need seed

    if model_name == 'CATB':
        if args.od_type == 'Iter':
            model = ctb.CatBoostClassifier(
                            custom_metric=['AUC','Accuracy'],
                            # eval_metric ='AUC',
                            iterations=args.n_epochs,
                            depth=args.depth,
                            learning_rate=args.lr,
                            verbose=args.verbose,
                            loss_function=args.LOSS_FUNCTION, #사용자 지정 로스도 가능한 모양
                            l2_leaf_reg = args.l2_leaf_reg,
                            od_type=args.od_type,
                            grow_policy=args.grow_policy,
                            task_type='GPU'
                            )

        else:
            model = ctb.CatBoostClassifier(
                            custom_metric=['AUC','Accuracy'],
                            # eval_metric ='AUC',
                            iterations=args.n_epochs,
                            depth=args.depth,
                            learning_rate=args.lr,
                            # verbose=args.verbose,
                            loss_function=args.LOSS_FUNCTION, #사용자 지정 로스도 가능한 모양
                            l2_leaf_reg = args.l2_leaf_reg,
                            od_type=args.od_type,
                            od_pval=0.05,
                            od_wait=50,
                            grow_policy=args.grow_policy,
                            task_type='GPU'
                            )
        
    return model