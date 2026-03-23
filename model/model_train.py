"""
模型训练模块
使用XGBoost进行模型训练和优化
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用相对路径（相对于model/文件夹）
# model/文件夹的父目录是项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')  # ../user_data
MODEL_DIR = os.path.join(BASE_DIR, 'model')  # ../model

# 尝试导入Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna未安装，将使用基础优化方法")


def train_model(use_advanced_optimization=True):
    """
    训练模型
    
    Args:
        use_advanced_optimization: 是否使用高级优化（Optuna）
    """
    print("="*60)
    print("【步骤3】模型训练")
    print("="*60)
    
    # 读取数据
    data_path = os.path.join(USER_DATA_DIR, 'train_data_feature_processed.csv')
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在: {data_path}")
        return False
    
    print(f"\n正在读取数据: {data_path}")
    df = pd.read_csv(data_path, sep=',')
    print(f"✅ 数据加载成功！数据维度: {df.shape}")
    
    # 准备特征和目标变量
    target_col = 'price'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"目标变量均值: {y.mean():.2f}")
    
    # 处理缺失值
    print(f"\n检查缺失值...")
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        print(f"⚠️ 发现缺失值的列: {missing_cols}")
        X = X.fillna(X.median())
        print("✅ 已用中位数填充缺失值")
    else:
        print("✅ 无缺失值")
    
    # 5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n🔄 使用 {n_splits} 折交叉验证进行模型调优...")
    
    # 读取之前的最优参数（如果存在）
    previous_best_params = None
    previous_best_path = os.path.join(MODEL_DIR, 'xgb_best_params.json')
    if os.path.exists(previous_best_path):
        with open(previous_best_path, 'r', encoding='utf-8') as f:
            previous_best_params = json.load(f)
        print(f"📋 发现之前的最优参数，将在此基础上进行优化")
    
    # 参数优化
    if use_advanced_optimization and OPTUNA_AVAILABLE:
        print(f"\n使用Optuna贝叶斯优化...")
        best_params = optimize_with_optuna(X, y, kf, previous_best_params)
    else:
        print(f"\n使用基础参数优化...")
        best_params = get_default_params(previous_best_params)
    
    # 使用最优参数进行5折交叉验证
    print(f"\n{'='*60}")
    print(f"📊 使用最优参数进行详细评估...")
    print(f"{'='*60}")
    
    final_params = {k: v for k, v in best_params.items() 
                    if k not in ['objective', 'eval_metric', 'tree_method']}
    final_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    })
    
    mae_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model = xgb.XGBRegressor(**final_params)
        try:
            from xgboost.callback import EarlyStopping
            callbacks = [EarlyStopping(rounds=100, save_best=True)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
                verbose=False
            )
        except (ImportError, AttributeError, TypeError):
            # 如果没有EarlyStopping，使用eval_set和verbose参数
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            except:
                model.fit(X_train, y_train, verbose=False)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算指标
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        print(f"  Fold {fold}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    
    # 计算平均指标
    avg_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    print(f"\n{'='*60}")
    print(f"📈 最终评估结果")
    print(f"{'='*60}")
    print(f"平均MAE: {avg_mae:.4f} (±{std_mae:.4f})")
    print(f"平均RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")
    
    # 使用全部数据训练最终模型
    print(f"\n{'='*60}")
    print(f"🚀 使用全部数据训练最终模型...")
    print(f"{'='*60}")
    
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X, y)
    
    print(f"✅ 最终模型训练完成！")
    
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, 'xgb_optimized_model.pkl')
    joblib.dump(final_model, model_save_path)
    print(f"✅ 模型已保存到: {model_save_path}")
    
    # 保存最优参数
    params_save_path = os.path.join(MODEL_DIR, 'xgb_best_params.json')
    with open(params_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_params, f, indent=2, ensure_ascii=False)
    print(f"✅ 最优参数已保存到: {params_save_path}")
    
    # 保存评估结果
    results = {
        'best_params': final_params,
        'avg_mae': float(avg_mae),
        'std_mae': float(std_mae),
        'avg_rmse': float(avg_rmse),
        'std_rmse': float(std_rmse),
        'mae_scores': [float(x) for x in mae_scores],
        'rmse_scores': [float(x) for x in rmse_scores],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimization_method': 'Optuna' if (use_advanced_optimization and OPTUNA_AVAILABLE) else 'Default'
    }
    
    results_save_path = os.path.join(MODEL_DIR, 'xgb_optimization_results.json')
    with open(results_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 评估结果已保存到: {results_save_path}")
    
    return True


def optimize_with_optuna(X, y, kf, previous_best_params=None):
    """使用Optuna进行参数优化"""
    def objective(trial):
        if previous_best_params:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 
                    max(200, previous_best_params['n_estimators'] - 100),
                    min(1000, previous_best_params['n_estimators'] + 200)),
                'max_depth': trial.suggest_int('max_depth',
                    max(4, previous_best_params['max_depth'] - 2),
                    min(10, previous_best_params['max_depth'] + 2)),
                'learning_rate': trial.suggest_float('learning_rate',
                    max(0.01, previous_best_params['learning_rate'] - 0.05),
                    min(0.2, previous_best_params['learning_rate'] + 0.05),
                    log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            }
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            }
        
        params.update({
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_bin': 256,
            'grow_policy': 'lossguide'
        })
        
        mae_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            try:
                from xgboost.callback import EarlyStopping
                callbacks = [EarlyStopping(rounds=100, save_best=True)]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                    verbose=False
                )
            except (ImportError, AttributeError, TypeError):
                try:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                except:
                    model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        return np.mean(mae_scores)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name='xgb_optimization'
    )
    
    if previous_best_params:
        try:
            initial_params = {
                'n_estimators': previous_best_params['n_estimators'],
                'max_depth': previous_best_params['max_depth'],
                'learning_rate': previous_best_params['learning_rate'],
                'subsample': previous_best_params['subsample'],
                'colsample_bytree': previous_best_params['colsample_bytree'],
                'min_child_weight': previous_best_params['min_child_weight'],
                'gamma': previous_best_params['gamma'],
                'reg_alpha': previous_best_params['reg_alpha'],
                'reg_lambda': previous_best_params['reg_lambda'],
            }
            study.enqueue_trial(initial_params)
        except Exception:
            pass
    
    n_trials = 100  # 增加试验次数以获得更好的参数
    print(f"⏳ 开始Optuna优化（{n_trials}次试验）...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    best_score = study.best_value
    
    print(f"\n✅ Optuna优化完成！")
    print(f"最优交叉验证MAE: {best_score:.4f}")
    
    return best_params


def get_default_params(previous_best_params=None):
    """获取默认参数"""
    if previous_best_params:
        return previous_best_params
    else:
        return {
            'n_estimators': 400,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
        }


if __name__ == '__main__':
    train_model(use_advanced_optimization=False)  # 默认不使用高级优化以加快速度

