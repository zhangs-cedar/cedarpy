dir=$(cd `dirname $0`; pwd)/
cd ${dir}
ruff format . # 规范化代码
rm -rf dist # && pip install --upgrade setuptools wheel # -i https://mirror.baidu.com/pypi/simple # 安装打包工具
# Check if Python 3 is installed
python setup.py sdist bdist_wheel
rm -rf cedar.egg-info && rm -rf build && rm -rf dist/*.tar.gz # 删除生成文件夹、文件
pip uninstall -y cedar && pip install dist/*.whl # 卸载再重新安装cedar
