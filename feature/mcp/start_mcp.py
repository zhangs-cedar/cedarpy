#!/usr/bin/env python3
"""OpenCV MCP服务启动脚本

这个脚本用于在Cursor中快速启动OpenCV MCP服务
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """启动MCP服务"""
    # 确保当前目录正确
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 检查依赖是否安装
    try:
        import cv2
        import numpy as np
        from mcp.server.fastmcp import FastMCP
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return 1
    
    # 启动智能OpenCV MCP服务
    print("🚀 启动智能OpenCV MCP服务...")
    try:
        subprocess.run([sys.executable, "smart_opencv_mcp.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 服务启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 