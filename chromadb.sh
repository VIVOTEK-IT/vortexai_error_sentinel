#!/bin/bash
# ChromaDB 命令行工具包装器

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查 Python 是否可用
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

# 检查 ChromaDB 数据目录
if [ ! -d "./data/chroma_db" ]; then
    echo -e "${RED}❌ ChromaDB 数据目录不存在: ./data/chroma_db${NC}"
    exit 1
fi

# 显示帮助信息
show_help() {
    echo -e "${BLUE}🔧 ChromaDB 命令行工具${NC}"
    echo "=" * 50
    echo ""
    echo -e "${YELLOW}交互式模式:${NC}"
    echo "  ./chromadb.sh                    # 启动交互式 CLI"
    echo ""
    echo -e "${YELLOW}快速命令:${NC}"
    echo "  ./chromadb.sh stats              # 查看统计信息"
    echo "  ./chromadb.sh search 'query'     # 搜索文档"
    echo "  ./chromadb.sh list [limit]       # 列出文档"
    echo "  ./chromadb.sh filter key value   # 过滤文档"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./chromadb.sh search 'database error'"
    echo "  ./chromadb.sh list 20"
    echo "  ./chromadb.sh filter service api-gateway"
    echo ""
}

# 主函数
main() {
    case "$1" in
        "stats")
            echo -e "${GREEN}📊 获取 ChromaDB 统计信息...${NC}"
            python3 quick_chromadb.py stats
            ;;
        "search")
            if [ -z "$2" ]; then
                echo -e "${RED}❌ 请提供搜索查询${NC}"
                echo "用法: ./chromadb.sh search 'your query'"
                exit 1
            fi
            echo -e "${GREEN}🔍 搜索: $2${NC}"
            python3 quick_chromadb.py search --query "$2" --limit "${3:-10}"
            ;;
        "list")
            echo -e "${GREEN}📄 列出文档...${NC}"
            python3 quick_chromadb.py list --limit "${2:-10}"
            ;;
        "filter")
            if [ -z "$2" ] || [ -z "$3" ]; then
                echo -e "${RED}❌ 请提供过滤条件${NC}"
                echo "用法: ./chromadb.sh filter key value"
                exit 1
            fi
            echo -e "${GREEN}🔍 过滤: $2 = $3${NC}"
            python3 quick_chromadb.py filter --key "$2" --value "$3" --limit "${4:-10}"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "")
            echo -e "${GREEN}🚀 启动 ChromaDB 交互式 CLI...${NC}"
            python3 chromadb_cli.py
            ;;
        *)
            echo -e "${RED}❌ 未知命令: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
