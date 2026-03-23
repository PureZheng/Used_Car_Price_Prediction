"""
特征工程模块
从处理后的数据中提取和构建特征
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用相对路径（相对于feature/文件夹）
# feature/文件夹的父目录是项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')  # ../user_data


def extract_date_features(date_col):
    """从日期字段提取特征"""
    date_str = date_col.astype(str)
    year = date_str.str[:4].astype(int)
    month = date_str.str[4:6].astype(int)
    day = date_str.str[6:8].astype(int)
    quarter = ((month - 1) // 3) + 1
    return year, month, day, quarter


def build_features(df_train=None, df_test=None):
    """
    构建特征
    
    Args:
        df_train: 训练数据DataFrame（包含price列）
        df_test: 测试数据DataFrame（可选）
    
    Returns:
        df_train_processed: 处理后的训练数据
        df_test_processed: 处理后的测试数据（如果提供了测试数据）
    """
    print("="*60)
    print("【步骤2】特征工程")
    print("="*60)
    
    if df_train is None:
        # 读取训练数据
        input_file = os.path.join(USER_DATA_DIR, 'train_data_processed.csv')
        if not os.path.exists(input_file):
            print(f"❌ 错误: 文件不存在: {input_file}")
            return None, None
        
        print(f"\n正在读取数据: {input_file}")
        df_train = pd.read_csv(input_file, sep=',')
        print(f"✅ 数据加载成功！数据维度: {df_train.shape}")
    
    df_processed = df_train.copy()
    
    # ==================== 1. 时间特征工程 ====================
    print(f"\n【2.1】时间特征工程...")
    
    # 处理 regDate（注册日期）
    if 'regDate' in df_processed.columns:
        reg_year, reg_month, reg_day, reg_quarter = extract_date_features(df_processed['regDate'])
        df_processed['regYear'] = reg_year
        df_processed['regMonth'] = reg_month
        df_processed['regQuarter'] = reg_quarter
        print(f"  ✅ 从regDate提取了: regYear, regMonth, regQuarter")
    
    # 处理 creatDate（上线日期）
    if 'creatDate' in df_processed.columns:
        creat_year, creat_month, creat_day, creat_quarter = extract_date_features(df_processed['creatDate'])
        df_processed['creatYear'] = creat_year
        df_processed['creatMonth'] = creat_month
        df_processed['creatQuarter'] = creat_quarter
        print(f"  ✅ 从creatDate提取了: creatYear, creatMonth, creatQuarter")
    
    # 计算车龄（从注册到上线的年数）
    if 'regDate' in df_processed.columns and 'creatDate' in df_processed.columns:
        reg_date = pd.to_datetime(df_processed['regDate'].astype(str), format='%Y%m%d', errors='coerce')
        creat_date = pd.to_datetime(df_processed['creatDate'].astype(str), format='%Y%m%d', errors='coerce')
        df_processed['carAge'] = (creat_date - reg_date).dt.days / 365.25
        df_processed['daysFromRegToCreat'] = (creat_date - reg_date).dt.days
        print(f"  ✅ 计算了车龄特征: carAge, daysFromRegToCreat")
    
    # ==================== 2. 分类特征编码 ====================
    print(f"\n【2.2】分类特征编码...")
    
    # 2.1 低基数分类特征 - 转换为整数
    low_cardinality_cats = ['bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in low_cardinality_cats:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('Int64')
            print(f"  ✅ {col} 已转换为整数类型")
    
    # 2.2 高基数分类特征 - 使用频率编码和目标编码
    high_cardinality_cats = ['model', 'brand', 'regionCode']
    
    # 频率编码
    for col in high_cardinality_cats:
        if col in df_processed.columns:
            freq_map = df_processed[col].value_counts().to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
            print(f"  ✅ {col} 添加了频率编码: {col}_freq")
    
    # 目标编码（使用K折交叉验证避免数据泄露）
    if 'price' in df_processed.columns:
        print(f"  📊 开始目标编码（使用5折交叉验证）...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in high_cardinality_cats:
            if col in df_processed.columns:
                df_processed[f'{col}_target_enc'] = 0.0
                
                for train_idx, val_idx in kf.split(df_processed):
                    train_data = df_processed.iloc[train_idx]
                    val_data = df_processed.iloc[val_idx]
                    
                    # 计算训练集的均值
                    target_mean = train_data.groupby(col)['price'].mean()
                    
                    # 应用到验证集
                    df_processed.loc[val_idx, f'{col}_target_enc'] = val_data[col].map(target_mean).fillna(df_processed['price'].mean())
                
                print(f"  ✅ {col} 添加了目标编码: {col}_target_enc")
    
    # ==================== 3. 交互特征 ====================
    print(f"\n【2.3】创建交互特征...")
    
    # 功率密度特征
    if 'power' in df_processed.columns and 'kilometer' in df_processed.columns:
        df_processed['power_per_km'] = df_processed['power'] / (df_processed['kilometer'] + 1)
        df_processed['power_km_interaction'] = df_processed['power'] * df_processed['kilometer']
        df_processed['power_squared'] = df_processed['power'] ** 2
        df_processed['kilometer_squared'] = df_processed['kilometer'] ** 2
        print(f"  ✅ 创建了功率-里程交互特征: power_per_km, power_km_interaction, power_squared, kilometer_squared")
    
    # 年均行驶里程
    if 'kilometer' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['km_per_year'] = df_processed['kilometer'] / (df_processed['carAge'] + 0.1)
        df_processed['km_per_year_log'] = np.log1p(df_processed['km_per_year'])
        print(f"  ✅ 创建了年均行驶里程: km_per_year, km_per_year_log")
    
    # 品牌-车型组合特征
    if 'brand' in df_processed.columns and 'model' in df_processed.columns:
        df_processed['brand_model'] = df_processed['brand'].astype(str) + '_' + df_processed['model'].astype(str)
        brand_model_freq = df_processed['brand_model'].value_counts().to_dict()
        df_processed['brand_model_freq'] = df_processed['brand_model'].map(brand_model_freq)
        df_processed = df_processed.drop(columns=['brand_model'])
        print(f"  ✅ 创建了品牌-车型组合特征: brand_model_freq")
    
    # 更多交互特征
    if 'power' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['power_per_age'] = df_processed['power'] / (df_processed['carAge'] + 0.1)
        print(f"  ✅ 创建了功率-车龄交互特征: power_per_age")
    
    if 'kilometer' in df_processed.columns and 'power' in df_processed.columns and 'carAge' in df_processed.columns:
        df_processed['km_power_age'] = df_processed['kilometer'] * df_processed['power'] / (df_processed['carAge'] + 0.1)
        print(f"  ✅ 创建了里程-功率-车龄组合特征: km_power_age")
    
    # 对数变换特征（对偏态分布有帮助）
    numeric_cols = ['power', 'kilometer']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[f'{col}_log'] = np.log1p(df_processed[col])
            df_processed[f'{col}_sqrt'] = np.sqrt(df_processed[col] + 1)
            print(f"  ✅ 创建了{col}的对数和平方根变换特征")
    
    # 功率分箱特征
    if 'power' in df_processed.columns:
        df_processed['power'] = df_processed['power'].fillna(df_processed['power'].median())
        df_processed['power'] = df_processed['power'].clip(lower=0, upper=600)
        power_bin = pd.cut(df_processed['power'], bins=[0, 100, 150, 200, 300, 600], 
                          labels=[0, 1, 2, 3, 4], include_lowest=True)
        df_processed['power_bin'] = power_bin.cat.codes.fillna(1).astype(int)
        print(f"  ✅ 创建了功率分箱特征: power_bin")
    
    # 里程分箱特征
    if 'kilometer' in df_processed.columns:
        df_processed['kilometer'] = df_processed['kilometer'].fillna(df_processed['kilometer'].median())
        df_processed['kilometer'] = df_processed['kilometer'].clip(lower=0, upper=100)
        kilometer_bin = pd.cut(df_processed['kilometer'], bins=[0, 5, 10, 15, 20, 25, 100], 
                              labels=[0, 1, 2, 3, 4, 5], include_lowest=True)
        df_processed['kilometer_bin'] = kilometer_bin.cat.codes.fillna(2).astype(int)
        print(f"  ✅ 创建了里程分箱特征: kilometer_bin")
    
    # ==================== 4. 统计特征 ====================
    print(f"\n【2.4】创建统计特征...")
    
    # 按品牌统计价格特征
    if 'brand' in df_processed.columns and 'price' in df_processed.columns:
        brand_stats = df_processed.groupby('brand')['price'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_price_min', 'brand_price_max']
        df_processed = df_processed.merge(brand_stats, on='brand', how='left')
        # 添加价格相对位置特征（仅在训练数据中可用）
        df_processed['price_vs_brand_mean'] = df_processed['price'] / (df_processed['brand_price_mean'] + 1)
        df_processed['price_vs_brand_median'] = df_processed['price'] / (df_processed['brand_price_median'] + 1)
        print(f"  ✅ 创建了品牌统计特征: brand_price_mean, brand_price_median, brand_price_std, brand_price_min, brand_price_max, price_vs_brand_mean, price_vs_brand_median")
    
    # 按车型统计价格特征
    if 'model' in df_processed.columns and 'price' in df_processed.columns:
        model_stats = df_processed.groupby('model')['price'].agg(['mean', 'median', 'std']).reset_index()
        model_stats.columns = ['model', 'model_price_mean', 'model_price_median', 'model_price_std']
        df_processed = df_processed.merge(model_stats, on='model', how='left')
        # 添加价格相对位置特征（仅在训练数据中可用）
        df_processed['price_vs_model_mean'] = df_processed['price'] / (df_processed['model_price_mean'] + 1)
        print(f"  ✅ 创建了车型统计特征: model_price_mean, model_price_median, model_price_std, price_vs_model_mean")
    
    # 按地区统计价格特征
    if 'regionCode' in df_processed.columns and 'price' in df_processed.columns:
        region_stats = df_processed.groupby('regionCode')['price'].agg(['mean', 'median', 'std']).reset_index()
        region_stats.columns = ['regionCode', 'region_price_mean', 'region_price_median', 'region_price_std']
        df_processed = df_processed.merge(region_stats, on='regionCode', how='left')
        # 添加价格相对位置特征（仅在训练数据中可用）
        df_processed['price_vs_region_mean'] = df_processed['price'] / (df_processed['region_price_mean'] + 1)
        print(f"  ✅ 创建了地区统计特征: region_price_mean, region_price_median, region_price_std, price_vs_region_mean")
    
    # 组合统计特征
    if 'brand' in df_processed.columns and 'model' in df_processed.columns and 'price' in df_processed.columns:
        brand_model_stats = df_processed.groupby(['brand', 'model'])['price'].agg(['mean', 'median']).reset_index()
        brand_model_stats.columns = ['brand', 'model', 'brand_model_price_mean', 'brand_model_price_median']
        df_processed = df_processed.merge(brand_model_stats, on=['brand', 'model'], how='left')
        print(f"  ✅ 创建了品牌-车型组合统计特征: brand_model_price_mean, brand_model_price_median")
    
    # v系列特征的统计特征
    v_features = [f'v_{i}' for i in range(15)]
    v_cols = [v for v in v_features if v in df_processed.columns]
    if len(v_cols) > 0:
        df_processed['v_sum'] = df_processed[v_cols].sum(axis=1)
        df_processed['v_mean'] = df_processed[v_cols].mean(axis=1)
        df_processed['v_std'] = df_processed[v_cols].std(axis=1)
        df_processed['v_max'] = df_processed[v_cols].max(axis=1)
        df_processed['v_min'] = df_processed[v_cols].min(axis=1)
        print(f"  ✅ 创建了v系列统计特征: v_sum, v_mean, v_std, v_max, v_min")
    
    # 保存处理后的数据
    output_file = os.path.join(USER_DATA_DIR, 'train_data_feature_processed.csv')
    df_processed.to_csv(output_file, sep=',', index=False)
    print(f"\n✅ 特征工程完成！")
    print(f"   处理前维度: {df_train.shape}")
    print(f"   处理后维度: {df_processed.shape}")
    print(f"   新增特征数: {df_processed.shape[1] - df_train.shape[1]}")
    print(f"   保存路径: {output_file}")
    
    return df_processed, None


if __name__ == '__main__':
    build_features()

