@echo off
:: ========== 核心配置 ==========
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
:: ==============================

echo ==============================
echo 🍵 饮料推荐模型训练脚本
echo 🐍 适配Python 3.12.4版本
echo ==============================
echo.

:: 1. 升级pip
echo 📌 正在升级pip...
python.exe -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --disable-pip-version-check >nul 2>&1

:: 2. 安装TensorFlow
echo 📦 安装TensorFlow...
pip install tensorflow==2.17.0 -i https://mirrors.aliyun.com/pypi/simple/ --disable-pip-version-check

:: 3. 解决protobuf冲突（关键！）
echo 🛠️ 修复依赖冲突...
pip install protobuf==4.25.3 --force-reinstall -i https://mirrors.aliyun.com/pypi/simple/ --disable-pip-version-check >nul 2>&1

:: 4. 安装其他依赖
echo 📦 安装其他依赖包...
pip install pandas scikit-learn numpy matplotlib joblib openpyxl -i https://mirrors.aliyun.com/pypi/simple/ --disable-pip-version-check >nul 2>&1

:: 5. 运行训练脚本（屏蔽警告）
echo.
echo 🚀 开始训练模型...
python -W ignore::UserWarning train_model.py

echo.
echo ==============================
echo 🎉 训练完成！
echo ==============================
pause >nul