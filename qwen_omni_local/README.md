# 本地运行 Qwen2.5-Omni 模型

这个目录包含了在本地运行 Qwen2.5-Omni 模型的代码和说明。

## 目录结构

- `requirements.txt`: Python 依赖项
- `inference.py`: 模型推理脚本
- `app.py`: Gradio 界面应用

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行推理

```bash
python inference.py
```

### 启动 Web 界面

```bash
python app.py
```

## 模型路径

模型文件位于 [/data/qwen_omni/](file:///data/qwen_omni/) 目录中，包含完整的模型权重和配置文件。

