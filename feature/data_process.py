"""
数据处理模块
处理原始训练数据，包括字段清理、异常值处理等
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 使用相对路径（相对于feature/文件夹）
DATA_DIR = os.path.join(BASE_DIR, 'data')  # ../data
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')  # ../user_data


def process_train_data():
    """处理训练数据"""
    print("="*60)
    print("【步骤1】数据处理")
    print("="*60)
    
    # 输入文件路径
    input_file = os.path.join(DATA_DIR, 'used_car_train_20200313.csv')
    output_file = os.path.join(USER_DATA_DIR, 'train_data_processed.csv')
    
    # 创建输出目录
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return False
    
    # 读取原始数据（使用空格分隔）
    print(f"\n正在读取数据: {input_file}")
    df = pd.read_csv(input_file, sep=' ')
    print(f"✅ 数据加载成功！数据维度: {df.shape}")
    
    # 保存原始数据副本
    original_file = os.path.join(USER_DATA_DIR, 'original_train_data.csv')
    df.to_csv(original_file, sep=',', index=False)
    print(f"✅ 原始数据已保存到: {original_file}")
    
    # 1. 删除无关字段
    print(f"\n【1.1】删除无关字段...")
    if 'SaleID' in df.columns:
        df = df.drop(columns=['SaleID'])
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    print(f"✅ 已删除SaleID和name字段")
    
    # 2. 处理notRepairedDamage字段
    print(f"\n【1.2】处理notRepairedDamage字段...")
    if 'notRepairedDamage' in df.columns:
        # 将"-"替换为NaN
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        # 转换为数值类型
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')
        print(f"✅ notRepairedDamage字段已处理")
    
    # 3. 处理creatDate字段（修复无效月份）
    print(f"\n【1.3】处理creatDate字段...")
    if 'creatDate' in df.columns:
        def fix_creatDate(x):
            if pd.isna(x):
                return x
            x_str = str(int(x)) if isinstance(x, (int, float)) else str(x)
            if len(x_str) == 8:
                month_str = x_str[4:6]
                if month_str not in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                    return int(x_str[:4] + '06' + x_str[6:])
            return int(x) if isinstance(x, (int, float)) else x
        
        df['creatDate'] = df['creatDate'].apply(fix_creatDate)
        print(f"✅ creatDate字段已处理")
    
    # 4. 处理v系列字段的异常值（使用IQR方法）
    print(f"\n【1.4】处理v系列字段异常值...")
    v_features = [f'v_{i}' for i in range(15)]
    for v in v_features:
        if v in df.columns:
            Q1 = df[v].quantile(0.25)
            Q3 = df[v].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # 替换离群值
            df.loc[df[v] < lower, v] = lower if lower > 0 else df[v].min()
            df.loc[df[v] > upper, v] = upper
    
    print(f"✅ v系列字段异常值已处理")
    
    # 5. 处理v_5, v_6, v_7, v_8, v_9字段（使用model+brand分组填充）
    print(f"\n【1.5】处理v_5-v_9字段...")
    v_fields_to_process = ['v_5', 'v_6', 'v_7', 'v_8', 'v_9']
    for v_field in v_fields_to_process:
        if v_field in df.columns and 'model' in df.columns and 'brand' in df.columns:
            # 计算每个model+brand组合的众数
            def get_mode_value(x):
                mode_series = x.mode()
                if len(mode_series) > 0:
                    return mode_series.iloc[0]
                return None
            
            mode_map = df[df[v_field] != 0].groupby(['model', 'brand'])[v_field].apply(get_mode_value)
            filled_count = 0
            
            for (model, brand), mode_value in mode_map.items():
                if mode_value is not None:
                    mask = (df['model'] == model) & (df['brand'] == brand) & (df[v_field] == 0)
                    count = mask.sum()
                    if count > 0:
                        df.loc[mask, v_field] = mode_value
                        filled_count += count
            
            print(f"  ✅ {v_field}: 填充了 {filled_count} 条记录")
    
    # 6. 删除v_6和v_7为0的行
    print(f"\n【1.6】删除v_6和v_7为0的行...")
    for v_field in ['v_6', 'v_7', 'v_9']:
        if v_field in df.columns:
            zero_mask = df[v_field] == 0
            count = zero_mask.sum()
            if count > 0:
                df = df[~zero_mask]
                print(f"  ✅ {v_field}: 删除了 {count} 行")
    
    # 保存处理后的数据
    print(f"\n正在保存处理后的数据...")
    df.to_csv(output_file, sep=',', index=False)
    print(f"✅ 数据已保存到: {output_file}")
    print(f"   处理后数据维度: {df.shape}")
    
    return True


if __name__ == '__main__':
    process_train_data()

