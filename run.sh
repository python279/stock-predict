#!/bin/bash

# 全球新闻分析系统启动脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}全球新闻分析系统${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python 3${NC}"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "Python 版本: ${GREEN}$PYTHON_VERSION${NC}"

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}错误: 配置文件 config.yaml 不存在${NC}"
    exit 1
fi

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}错误: requirements.txt 不存在${NC}"
    exit 1
fi

# 询问是否安装依赖
read -p "是否需要安装/更新依赖包? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}正在安装依赖...${NC}"
    .venv/bin/pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}依赖安装成功${NC}"
    else
        echo -e "${RED}依赖安装失败${NC}"
        exit 1
    fi
fi

# 创建必要的目录
mkdir -p data/news_cache data/reports logs

echo ""
echo "请选择运行模式:"
echo "  1) 立即执行一次"
echo "  2) 启动定时调度器 (后台运行)"
echo "  3) 启动定时调度器 (前台运行)"
echo "  4) 停止后台调度器"
echo "  5) 查看日志"
read -p "请选择 (1-5): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo -e "${YELLOW}立即执行任务...${NC}"
        .venv/bin/python3 src/main.py
        ;;
    2)
        echo -e "${YELLOW}启动后台调度器...${NC}"
        nohup .venv/bin/python3 src/scheduler.py > logs/scheduler.log 2>&1 &
        PID=$!
        echo $PID > logs/scheduler.pid
        echo -e "${GREEN}调度器已在后台启动 (PID: $PID)${NC}"
        echo "使用以下命令查看日志:"
        echo "  tail -f logs/scheduler.log"
        echo "  tail -f logs/news_analyzer.log"
        ;;
    3)
        echo -e "${YELLOW}启动前台调度器...${NC}"
        .venv/bin/python3 src/scheduler.py
        ;;
    4)
        if [ -f "logs/scheduler.pid" ]; then
            PID=$(cat logs/scheduler.pid)
            echo -e "${YELLOW}正在停止调度器 (PID: $PID)...${NC}"
            kill $PID 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}调度器已停止${NC}"
                rm logs/scheduler.pid
            else
                echo -e "${RED}未找到运行中的调度器进程${NC}"
                rm -f logs/scheduler.pid
            fi
        else
            echo -e "${RED}未找到调度器 PID 文件${NC}"
        fi
        ;;
    5)
        echo -e "${YELLOW}查看日志 (Ctrl+C 退出)...${NC}"
        if [ -f "logs/news_analyzer.log" ]; then
            tail -f logs/news_analyzer.log
        else
            echo -e "${RED}日志文件不存在${NC}"
        fi
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}完成！${NC}"

