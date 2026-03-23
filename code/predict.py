"""
预测模块
对测试数据进行预测
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录和feature文件夹到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'feature'))

# 导入特征工程模块
from feature.feature_engineering import extract_date_features

# 使用相对路径（相对于code/文件夹）
# code/文件夹的父目录是项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # ../data
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')  # ../user_data
MODEL_DIR = os.path.join(BASE_DIR, 'model')  # ../model
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_result')  # ../prediction_result


def predict_test_data():
    """对测试数据进行预测"""
    print("="*60)
    print("【步骤4】预测")
    print("="*60)
    
    # 文件路径
    test_data_path = os.path.join(DATA_DIR, 'used_car_testA_20200313.csv')
    train_data_path = os.path.join(USER_DATA_DIR, 'train_data_feature_processed.csv')
    model_path = os.path.join(MODEL_DIR, 'xgb_optimized_model.pkl')
    output_path = os.path.join(PREDICTION_DIR, 'predictions.csv')
    
    # 创建输出目录
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # ==================== 1. 读取测试数据 ====================
    print(f"\n【4.1】读取测试数据...")
    if not os.path.exists(test_data_path):
        print(f"❌ 错误: 测试数据文件不存在: {test_data_path}")
        return False
    
    # 测试数据使用空格分隔
    df_test = pd.read_csv(test_data_path, sep=' ')
    print(f"✅ 测试数据加载成功！数据维度: {df_test.shape}")
    
    # 保存SaleID用于最终输出
    if 'SaleID' in df_test.columns:
        sale_ids = df_test['SaleID'].copy()
    else:
        print("⚠️ 警告: 未找到SaleID列，将使用索引作为ID")
        sale_ids = df_test.index
    
    # ==================== 2. 读取训练数据（用于获取编码映射和统计信息） ====================
    print(f"\n【4.2】读取训练数据（用于特征映射）...")
    if not os.path.exists(train_data_path):
        print(f"❌ 错误: 训练数据文件不存在: {train_data_path}")
        return False
    
    df_train = pd.read_csv(train_data_path, sep=',')
    print(f"✅ 训练数据加载成功！数据维度: {df_train.shape}")
    
    # ==================== 3. 数据处理 ====================
    print(f"\n【4.3】数据处理...")
    df_processed = df_test.copy()
    
    # 3.1 处理notRepairedDamage字段
    if 'notRepairedDamage' in df_processed.columns:
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', np.nan)
        df_processed['notRepairedDamage'] = pd.to_numeric(df_processed['notRepairedDamage'], errors='coerce')
        print(f"  ✅ 处理了notRepairedDamage字段")
    
    # 3.2 处理creatDate字段
    if 'creatDate' in df_processed.columns:
        def fix_creatDate(x):
            if pd.isna(x):
                return x
            x_str = str(int(x)) if isinstance(x, (int, float)) else str(x)
            if len(x_str) == 8:
                month_str = x_str[4:6]
                if month_str not in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                    return int(x_str[:4] + '06' + x_str[6:])
            return int(x) if isinstance(x, (int, float)) else x
        
        df_processed['creatDate'] = df_processed['creatDate'].apply(fix_creatDate)
        print(f"  ✅ 处理了creatDate字段")
    
    # 3.3 处理v系列字段的异常值
    v_features = [f'v_{i}' for i in range(15)]
    for v in v_features:
        if v in df_processed.columns:
            Q1 = df_processed[v].quantile(0.25)
            Q3 = df_processed[v].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            df_processed.loc[df_processed[v] < lower, v] = lower if lower > 0 else df_processed[v].min()
            df_processed.loc[df_processed[v] > upper, v] = upper
    
    print(f"  ✅ 处理了v系列字段异常值")
    
    # ==================== 4. 特征工程 ====================
    print(f"\n【4.4】特征工程...")
    
    # 4.1 时间特征工程
    if 'regDate' in df_processed.columns:
        reg_year, reg_month, reg_day, reg_quarter = extract_date_features(df_processed['regDate'])
        df_processed['regYear'] = reg_year
        df_processed['regMonth'] = reg_month
        df_processed['regQuarter'] = reg_quarter
        print(f"  ✅ 从regDate提取了: regYear, regMonth, regQuarter")
    
    if 'creatDate' in df_processed.columns:
        creat_year, creat_month, creat_day, creat_quarter = extract_date_features(df_processed['creatDate'])
        df_processed['creatYear'] = creat_year
        df_processed['creatMonth'] = creat_month
        df_processed['creatQuarter'] = creat_quarter
        print(f"  ✅ 从creatDate提取了: creatYear, creatMonth, creatQuarter")
    
    if 'regDate' in df_processed.columns and 'creatDate' in df_processed.columns:
        reg_date = pd.to_datetime(df_processed['regDate'].astype(str), format='%Y%m%d', errors='coerce')
        creat_date = pd.to_datetime(df_processed['creatDate'].astype(str), format='%Y%m%d', errors='coerce')
        df_processed['carAge'] = (creat_date - reg_date).dt.days / 365.25
        df_processed['daysFromRegToCreat'] = (creat_date - reg_date).dt.days
        print(f"  ✅ 计算了车龄特征: carAge, daysFromRegToCreat")
    
    # 4.2 分类特征编码
    print(f"  📊 处理分类特征编码...")
    
    # 低基数分类特征
    low_cardinality_cats = ['bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in low_cardinality_cats:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('Int64')
            print(f"    ✅ {col} 已转换为整数类型")
    
    # 高基数分类特征 - 使用训练数据的频率编码
    high_cardinality_cats = ['model', 'brand', 'regionCode']
    for col in high_cardinality_cats:
        if col in df_processed.columns:
            freq_map = df_train[col].value_counts().to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map).fillna(0)
            print(f"    ✅ {col} 添加了频率编码: {col}_freq")
    
    # 高基数分类特征 - 使用训练数据的目标编码
    if 'price' in df_train.columns:
        for col in high_cardinality_cats:
            if col in df_processed.columns:
                target_mean = df_train.groupby(col)['price'].mean().to_dict()
                default_value = df_train['price'].mean()
                df_processed[f'{col}_target_enc'] = df_processed[col].map(target_mean).fillna(default_value)
                print(f"    ✅ {col} 添加了目标编码: {col}_target_enc")
    
    # 4.3 交互特征
    print(f"  📊 创建交互特征...")
    
    if 'power' in df_processed.columns and 'kilometer' in df_processed.columns:
        df_processed['power_per_km'] = df_processed['power'] / (df_processed['kilometer'] + 1)
        df_processed['power_km_interaction'] = df_processed['power'] * df_processed['kilometer']
        df_processed['power_squared'] = df_processed['power'] ** 2
        df_processed['kilometer_squared'] = df_processed['kilometer'] ** 2
        print(f"    ✅ 创建了功率-里程交互特征")
    
    if 'kilometer' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['km_per_year'] = df_processed['kilometer'] / (df_processed['carAge'] + 0.1)
        df_processed['km_per_year_log'] = np.log1p(df_processed['km_per_year'])
        print(f"    ✅ 创建了年均行驶里程")
    
    # 更多交互特征
    if 'power' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['power_per_age'] = df_processed['power'] / (df_processed['carAge'] + 0.1)
        print(f"    ✅ 创建了功率-车龄交互特征")
    
    if 'kilometer' in df_processed.columns and 'power' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['km_power_age'] = df_processed['kilometer'] * df_processed['power'] / (df_processed['carAge'] + 0.1)
        print(f"    ✅ 创建了里程-功率-车龄组合特征")
    
    # 对数变换特征
    numeric_cols = ['power', 'kilometer']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[f'{col}_log'] = np.log1p(df_processed[col])
            df_processed[f'{col}_sqrt'] = np.sqrt(df_processed[col] + 1)
            print(f"    ✅ 创建了{col}的对数和平方根变换特征")
    
    if 'brand' in df_processed.columns and 'model' in df_processed.columns:
        df_processed['brand_model'] = df_processed['brand'].astype(str) + '_' + df_processed['model'].astype(str)
        brand_model_freq_map = (df_train['brand'].astype(str) + '_' + df_train['model'].astype(str)).value_counts().to_dict()
        df_processed['brand_model_freq'] = df_processed['brand_model'].map(brand_model_freq_map).fillna(0)
        df_processed = df_processed.drop(columns=['brand_model'])
        print(f"    ✅ 创建了品牌-车型组合特征")
    
    if 'power' in df_processed.columns:
        power_median = df_processed['power'].median() if not df_processed['power'].isna().all() else 122
        df_processed['power'] = df_processed['power'].fillna(power_median)
        df_processed['power'] = df_processed['power'].clip(lower=0, upper=600)
        power_bin = pd.cut(df_processed['power'], bins=[0, 100, 150, 200, 300, 600], 
                          labels=[0, 1, 2, 3, 4], include_lowest=True)
        df_processed['power_bin'] = power_bin.cat.codes.fillna(1).astype(int)
        print(f"    ✅ 创建了功率分箱特征")
    
    if 'kilometer' in df_processed.columns:
        kilometer_median = df_processed['kilometer'].median() if not df_processed['kilometer'].isna().all() else 15
        df_processed['kilometer'] = df_processed['kilometer'].fillna(kilometer_median)
        df_processed['kilometer'] = df_processed['kilometer'].clip(lower=0, upper=100)
        kilometer_bin = pd.cut(df_processed['kilometer'], bins=[0, 5, 10, 15, 20, 25, 100], 
                              labels=[0, 1, 2, 3, 4, 5], include_lowest=True)
        df_processed['kilometer_bin'] = kilometer_bin.cat.codes.fillna(2).astype(int)
        print(f"    ✅ 创建了里程分箱特征")
    
    # 4.4 统计特征
    print(f"  📊 创建统计特征...")
    
    if 'brand' in df_processed.columns and 'price' in df_train.columns:
        brand_stats = df_train.groupby('brand')['price'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_price_min', 'brand_price_max']
        df_processed = df_processed.merge(brand_stats, on='brand', how='left')
        df_processed['brand_price_mean'] = df_processed['brand_price_mean'].fillna(df_train['price'].mean())
        df_processed['brand_price_median'] = df_processed['brand_price_median'].fillna(df_train['price'].median())
        df_processed['brand_price_std'] = df_processed['brand_price_std'].fillna(df_train['price'].std())
        df_processed['brand_price_min'] = df_processed['brand_price_min'].fillna(df_train['price'].min())
        df_processed['brand_price_max'] = df_processed['brand_price_max'].fillna(df_train['price'].max())
        # 注意：测试数据没有price，所以这些相对特征需要基于训练数据的统计值
        print(f"    ✅ 创建了品牌统计特征")
    
    if 'model' in df_processed.columns and 'price' in df_train.columns:
        model_stats = df_train.groupby('model')['price'].agg(['mean', 'median', 'std']).reset_index()
        model_stats.columns = ['model', 'model_price_mean', 'model_price_median', 'model_price_std']
        df_processed = df_processed.merge(model_stats, on='model', how='left')
        df_processed['model_price_mean'] = df_processed['model_price_mean'].fillna(df_train['price'].mean())
        df_processed['model_price_median'] = df_processed['model_price_median'].fillna(df_train['price'].median())
        df_processed['model_price_std'] = df_processed['model_price_std'].fillna(df_train['price'].std())
        print(f"    ✅ 创建了车型统计特征")
    
    if 'regionCode' in df_processed.columns and 'price' in df_train.columns:
        region_stats = df_train.groupby('regionCode')['price'].agg(['mean', 'median', 'std']).reset_index()
        region_stats.columns = ['regionCode', 'region_price_mean', 'region_price_median', 'region_price_std']
        df_processed = df_processed.merge(region_stats, on='regionCode', how='left')
        df_processed['region_price_mean'] = df_processed['region_price_mean'].fillna(df_train['price'].mean())
        df_processed['region_price_median'] = df_processed['region_price_median'].fillna(df_train['price'].median())
        df_processed['region_price_std'] = df_processed['region_price_std'].fillna(df_train['price'].std())
        print(f"    ✅ 创建了地区统计特征")
    
    # 组合统计特征
    if 'brand' in df_processed.columns and 'model' in df_processed.columns and 'price' in df_train.columns:
        brand_model_stats = df_train.groupby(['brand', 'model'])['price'].agg(['mean', 'median']).reset_index()
        brand_model_stats.columns = ['brand', 'model', 'brand_model_price_mean', 'brand_model_price_median']
        df_processed = df_processed.merge(brand_model_stats, on=['brand', 'model'], how='left')
        df_processed['brand_model_price_mean'] = df_processed['brand_model_price_mean'].fillna(df_train['price'].mean())
        df_processed['brand_model_price_median'] = df_processed['brand_model_price_median'].fillna(df_train['price'].median())
        print(f"    ✅ 创建了品牌-车型组合统计特征")
    
    # v系列特征的统计特征
    v_features = [f'v_{i}' for i in range(15)]
    v_cols = [v for v in v_features if v in df_processed.columns]
    if len(v_cols) > 0:
        df_processed['v_sum'] = df_processed[v_cols].sum(axis=1)
        df_processed['v_mean'] = df_processed[v_cols].mean(axis=1)
        df_processed['v_std'] = df_processed[v_cols].std(axis=1)
        df_processed['v_max'] = df_processed[v_cols].max(axis=1)
        df_processed['v_min'] = df_processed[v_cols].min(axis=1)
        print(f"    ✅ 创建了v系列统计特征")
    
    # ==================== 5. 准备特征 ====================
    print(f"\n【4.5】准备特征...")
    
    # 获取训练数据中的特征列（排除price和SaleID）
    train_feature_cols = [col for col in df_train.columns if col not in ['price', 'SaleID']]
    
    # 确保测试数据包含所有需要的特征
    missing_features = set(train_feature_cols) - set(df_processed.columns)
    if missing_features:
        print(f"⚠️ 警告: 测试数据缺少以下特征: {missing_features}")
        for feat in missing_features:
            if feat in df_train.columns:
                default_value = df_train[feat].median() if df_train[feat].dtype in ['int64', 'float64'] else 0
                df_processed[feat] = default_value
                print(f"    ✅ 为缺失特征 {feat} 填充默认值: {default_value}")
    
    # 选择特征列（按照训练数据的顺序）
    X_test = df_processed[train_feature_cols].copy()
    
    # 处理缺失值
    print(f"  检查缺失值...")
    missing_cols = X_test.columns[X_test.isnull().any()].tolist()
    if missing_cols:
        print(f"  ⚠️ 发现缺失值的列: {missing_cols}")
        for col in missing_cols:
            if col in df_train.columns:
                fill_value = df_train[col].median() if df_train[col].dtype in ['int64', 'float64'] else df_train[col].mode()[0] if len(df_train[col].mode()) > 0 else 0
                X_test[col] = X_test[col].fillna(fill_value)
        print(f"  ✅ 已填充缺失值")
    else:
        print(f"  ✅ 无缺失值")
    
    print(f"  ✅ 特征准备完成！特征数量: {len(X_test.columns)}")
    
    # ==================== 6. 加载模型并预测 ====================
    print(f"\n【4.6】加载模型并预测...")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return False
    
    model = joblib.load(model_path)
    print(f"✅ 模型加载成功！")
    
    # 进行预测
    print(f"  ⏳ 正在进行预测...")
    predictions = model.predict(X_test)
    print(f"  ✅ 预测完成！预测样本数: {len(predictions)}")
    
    # 确保预测值为正数（价格不能为负）
    predictions = np.maximum(predictions, 0)
    
    print(f"\n预测结果统计:")
    print(f"  最小值: {predictions.min():.2f}")
    print(f"  最大值: {predictions.max():.2f}")
    print(f"  平均值: {predictions.mean():.2f}")
    print(f"  中位数: {np.median(predictions):.2f}")
    
    # ==================== 7. 保存预测结果 ====================
    print(f"\n【4.7】保存预测结果...")
    
    # 创建提交文件
    submit_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': predictions
    })
    
    submit_df.to_csv(output_path, index=False, sep=',')
    print(f"✅ 预测结果已保存到: {output_path}")
    print(f"   预测样本数: {len(submit_df)}")
    
    print(f"\n{'='*60}")
    print(f"✨ 预测流程完成！")
    print(f"{'='*60}")
    
    return True


if __name__ == '__main__':
    predict_test_data()

