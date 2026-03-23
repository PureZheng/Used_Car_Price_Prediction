#!/bin/bash

# 二手车价格预测模型 - 主程序入口脚本
# 自动检测并使用可用的 Python 命令

# 检测 Python 命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python 命令，请先安装 Python 3.7+"
    exit 1
fi

echo "使用 Python 命令: $PYTHON_CMD"
echo "Python 版本:"
$PYTHON_CMD --version
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查依赖是否安装
echo "检查依赖..."
if ! $PYTHON_CMD -c "import pandas, numpy, sklearn, xgboost" 2>/dev/null; then
    echo "⚠️ 警告: 部分依赖未安装，正在尝试安装..."
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt
    else
        echo "❌ 错误: requirements.txt 文件不存在"
        exit 1
    fi
fi

echo ""
echo "开始运行主程序..."
echo ""

# 运行主程序
$PYTHON_CMD main.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 程序执行成功！"
    exit 0
else
    echo ""
    echo "❌ 程序执行失败！"
    exit 1
fi

