# 二手车价格预测模型

## 项目简介

本项目是一个基于机器学习的二手车价格预测系统，使用XGBoost算法对二手车价格进行预测。项目包含完整的数据处理、特征工程、模型训练和预测流程。

## 项目结构

```
project
    |--README.md
    |--data/
    |   |--used_car_train_20200313.csv
    |   |--used_car_testA_20200313.csv
    |   |--used_car_sample_submit.csv
    |--user_data/
    |   |--train_data_processed.csv
    |   |--train_data_feature_processed.csv
    |   |--original_train_data.csv
    |--feature/
    |   |--data_process.py          # 数据处理代码
    |   |--feature_engineering.py   # 特征工程代码
    |--model/
    |   |--model_train.py           # 模型训练代码
    |   |--xgb_optimized_model.pkl
    |   |--xgb_best_params.json
    |   |--xgb_optimization_results.json
    |--prediction_result/
    |   |--predictions.csv
    |--code/
    |   |--main.py                  # 主程序入口
    |   |--main.sh                  # Shell脚本入口
    |   |--requirements.txt
    |   |--predict.py               # 预测代码
```

## 解决方案及算法介绍

### 整体流程

1. **数据处理** (`feature/data_process.py`)
   - 删除无关字段（SaleID、name）
   - 处理缺失值和异常值
   - 修复日期字段异常（如creatDate无效月份）
   - 使用IQR方法处理v系列字段的离群值
   - 基于model+brand分组填充v_5-v_9字段的缺失值
   - 删除v_6、v_7、v_9为0的异常行

2. **特征工程** (`feature/feature_engineering.py`)
   - **时间特征**：从regDate和creatDate提取年份、月份、季度，计算车龄
   - **分类特征编码**：
     - 低基数特征（bodyType、fuelType、gearbox、notRepairedDamage）：Label Encoding
     - 高基数特征（model、brand、regionCode）：频率编码 + 目标编码（使用K折交叉验证避免数据泄露）
   - **交互特征**：
     - 功率-里程交互特征（power_per_km、power_km_interaction）
     - 年均行驶里程（km_per_year）
     - 品牌-车型组合特征（brand_model_freq）
     - 功率和里程的分箱特征
   - **统计特征**：
     - 按品牌、车型、地区统计价格均值、中位数、标准差

3. **模型训练** (`model/model_train.py`)
   - 使用XGBoost回归模型
   - 5折交叉验证进行模型评估
   - 支持Optuna贝叶斯优化进行超参数调优
   - 使用早停策略防止过拟合
   - 保存最优模型和参数

4. **预测** (`code/predict.py`)
   - 对测试数据进行相同的数据处理和特征工程
   - 使用训练好的模型进行预测
   - 确保预测值为非负数
   - 输出预测结果到prediction_result/predictions.csv

### 算法详情

#### XGBoost模型

XGBoost（Extreme Gradient Boosting）是一种基于梯度提升的集成学习算法，具有以下优势：
- 高效处理大规模数据
- 自动处理缺失值
- 内置正则化防止过拟合
- 支持并行计算

#### 超参数优化

- **默认参数**：适用于快速训练和测试
- **Optuna优化**：使用TPE（Tree-structured Parzen Estimator）采样器进行贝叶斯优化，自动搜索最优超参数组合

主要超参数包括：
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `learning_rate`: 学习率
- `subsample`: 样本采样比例
- `colsample_bytree`: 特征采样比例
- `min_child_weight`: 叶子节点最小权重
- `gamma`: 最小损失减少量
- `reg_alpha`: L1正则化系数
- `reg_lambda`: L2正则化系数

## 代码运行说明

### 环境要求

- Python 3.7+
- 依赖包见 `code/requirements.txt`

### 安装依赖

```bash
cd code
pip install -r requirements.txt
```

### 运行方式

#### 方式1：使用main.sh（推荐，自动检测Python命令）

直接运行主脚本，将自动检测并使用可用的Python命令：

```bash
cd code
bash main.sh
```

或者（如果已设置执行权限）：

```bash
cd code
./main.sh
```

#### 方式2：使用main.py

直接运行主程序，将自动执行完整流程：

```bash
cd code
python3 main.py
```

**注意**：在 macOS 和部分 Linux 系统上，需要使用 `python3` 而不是 `python`。

#### 方式2：分步骤运行

如果需要分步骤执行，可以单独运行各个模块：

```bash
cd code

# 步骤1：数据处理
python data_process.py

# 步骤2：特征工程
python feature_engineering.py

# 步骤3：模型训练
python model_train.py

# 步骤4：预测
python predict.py
```

### 输出文件说明

- `user_data/train_data_processed.csv`: 处理后的训练数据
- `user_data/train_data_feature_processed.csv`: 特征工程后的训练数据
- `model/xgb_optimized_model.pkl`: 训练好的模型文件
- `model/xgb_best_params.json`: 最优超参数
- `model/xgb_optimization_results.json`: 模型评估结果
- `prediction_result/predictions.csv`: 最终预测结果（包含SaleID和price两列）

### 注意事项

1. **Python版本**：需要 Python 3.7+，在 macOS 上通常使用 `python3` 命令
2. **数据路径**：所有代码使用相对路径，确保在 `code/` 目录下运行
3. **数据文件**：确保 `data/` 文件夹中包含训练数据和测试数据文件
4. **模型训练时间**：使用Optuna优化时可能需要较长时间，默认使用基础参数以加快速度
5. **内存要求**：处理大规模数据时可能需要较大内存，建议至少8GB
6. **预测结果**：预测结果会自动保存到 `prediction_result/predictions.csv`，格式与竞赛提交要求一致
7. **执行权限**：如果使用 `./main.sh`，需要先设置执行权限：`chmod +x main.sh`

## 模型性能

模型使用5折交叉验证进行评估，主要指标：
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Squared Error)**: 均方根误差

具体性能指标保存在 `model/xgb_optimization_results.json` 文件中。

## 技术栈

- **数据处理**: pandas, numpy
- **特征工程**: scikit-learn
- **模型训练**: XGBoost
- **超参数优化**: Optuna (可选)
- **模型评估**: scikit-learn

## 作者

二手车价格预测项目

## 更新日志

- 2024: 初始版本，完成基础功能
- 优化代码结构，符合竞赛提交规范

