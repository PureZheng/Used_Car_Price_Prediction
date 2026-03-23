"""
主程序入口
执行完整的预测流程：数据处理 -> 特征工程 -> 模型训练 -> 预测
"""
import os
import sys

# 添加项目根目录到路径，以便导入feature和model文件夹中的模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'feature'))
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))

# 导入各个模块
from feature.data_process import process_train_data
from feature.feature_engineering import build_features
from model.model_train import train_model
from predict import predict_test_data


def main():
    """主函数：执行完整的预测流程"""
    print("="*60)
    print("二手车价格预测模型 - 完整流程")
    print("="*60)
    print("\n流程说明:")
    print("  1. 数据处理：清理和预处理原始数据")
    print("  2. 特征工程：提取和构建特征")
    print("  3. 模型训练：训练XGBoost模型")
    print("  4. 预测：对测试数据进行预测")
    print("="*60)
    
    # 步骤1：数据处理
    print("\n")
    if not process_train_data():
        print("❌ 数据处理失败，程序退出")
        return False
    
    # 步骤2：特征工程
    print("\n")
    df_train_processed, _ = build_features()
    if df_train_processed is None:
        print("❌ 特征工程失败，程序退出")
        return False
    
    # 步骤3：模型训练
    print("\n")
    # 启用Optuna优化以获得更好的MAE
    if not train_model(use_advanced_optimization=True):
        print("❌ 模型训练失败，程序退出")
        return False
    
    # 步骤4：预测
    print("\n")
    if not predict_test_data():
        print("❌ 预测失败，程序退出")
        return False
    
    print("\n" + "="*60)
    print("✨ 所有流程完成！")
    print("="*60)
    print("\n输出文件:")
    print("  - 预测结果: ../prediction_result/predictions.csv")
    print("  - 训练模型: ../model/xgb_optimized_model.pkl")
    print("  - 模型参数: ../model/xgb_best_params.json")
    print("="*60)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
