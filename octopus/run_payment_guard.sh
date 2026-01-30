#!/bin/bash
# PaymentGuard Demo 独立运行脚本
# 所有运行文件均在 octopus 下，不依赖 CLLM 目录；可从 octopus 或项目根执行

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PaymentGuard Demo"
echo "=========================================="
echo "项目根: $PROJECT_ROOT"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi
echo "Python: $(python3 --version)"
echo ""

# 安装完整流水线依赖（复杂模式需 openai；可选用 requirements.txt）
python3 -c "import pandas, numpy, sklearn" 2>/dev/null || {
    echo "安装基础依赖: pandas numpy scikit-learn"
    pip3 install pandas numpy scikit-learn
}
python3 -c "import openai" 2>/dev/null || {
    echo "安装 openai（完整 CLLM 流水线必需）"
    pip3 install openai
}
REQ="$SCRIPT_DIR/requirements.txt"
if [ -f "$REQ" ]; then
    echo "检查 requirements: $REQ"
    pip3 install -q -r "$REQ" 2>/dev/null || true
fi
echo ""

python3 -m octopus.demo_payment_guard "$@"

echo ""
echo "=========================================="
echo "Demo 结束"
echo "=========================================="
if [ -f "$SCRIPT_DIR/paymentguard_report.md" ]; then
    echo "报告: $SCRIPT_DIR/paymentguard_report.md"
fi
if [ -f "$SCRIPT_DIR/paymentguard_report.html" ]; then
    echo "展示报告（用浏览器打开）: $SCRIPT_DIR/paymentguard_report.html"
fi
