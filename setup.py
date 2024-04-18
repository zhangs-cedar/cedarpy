import setuptools


setuptools.setup(
    name="cedar",  # 模块名称
    version="0.0.4",  # 当前版本
    author="zhangsong",  # 作者
    author_email="",  # 作者邮箱
    description="",  # 模块简介ma
    long_description="",  # 模块详细介绍
    long_description_content_type="",  # 模块详细介绍格式
    packages=setuptools.find_packages(),  # 自动找到项目中导入的模块
    package_data={
        "cedar.draw": ["simsun.ttc"],
    },
    # 模块相关的元数据
    classifiers=[],
    # 依赖模块
    install_requires=[
        # "numpy",
    ],
    python_requires=">=3.6",
)
