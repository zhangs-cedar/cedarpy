dir=$(cd `dirname $0`; pwd)/
cd ${dir}
sh black.sh # 规范化代码
rm -rf dist # && pip install --upgrade setuptools wheel # -i https://mirror.baidu.com/pypi/simple # 安装打包工具
# Check if Python 3 is installed
if command -v python3 &> /dev/null; then
    PYTHON_EXECUTABLE=python3
else
    PYTHON_EXECUTABLE=python
fi
# Run setup.py with the determined Python executable

$PYTHON_EXECUTABLE setup.py sdist bdist_wheel
rm -rf cedar.egg-info && rm -rf build && rm -rf dist/*.tar.gz # 删除生成文件夹、文件
pip uninstall -y cedar && pip install dist/*.whl # 卸载再重新安装cedar
