import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import gradio as gr
import os
from typing import Optional, Tuple, List, Any

# 模型路径
MODEL_PATH = "/data/qwen_omni/"

# 全局变量存储模型
model: Optional[Any] = None
tokenizer: Optional[Any] = None
processor: Optional[Any] = None

def load_model() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    加载 Qwen2.5-Omni 模型
    """
    global model, tokenizer, processor
    
    if model is not None:
        return model, tokenizer, processor
    
    print("正在加载模型...")
    try:
        # 尝试多种方式加载分词器，解决兼容性问题
        tokenizer = None
        # 方法1: 使用AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        except Exception as e:
            print(f"使用AutoTokenizer加载失败: {e}")
            
        # 方法2: 如果方法1失败，尝试直接指定模型类型
        if tokenizer is None:
            try:
                from transformers import Qwen2Tokenizer
                tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            except Exception as e:
                print(f"使用Qwen2Tokenizer加载失败: {e}")
                
        # 方法3: 如果前两种方法都失败，尝试使用模型配置文件中的信息
        if tokenizer is None:
            try:
                # 直接从配置文件加载分词器
                tokenizer_config_path = os.path.join(MODEL_PATH, "tokenizer_config.json")
                if os.path.exists(tokenizer_config_path):
                    import json
                    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                        tokenizer_config = json.load(f)
                    
                    # 根据配置文件中的信息选择合适的tokenizer类
                    tokenizer_class = tokenizer_config.get("tokenizer_class", "Qwen2Tokenizer")
                    if tokenizer_class:
                        tokenizer_module = __import__('transformers', fromlist=[tokenizer_class])
                        tokenizer_cls = getattr(tokenizer_module, tokenizer_class)
                        tokenizer = tokenizer_cls.from_pretrained(MODEL_PATH, trust_remote_code=True)
            except Exception as e:
                print(f"通过配置文件加载tokenizer失败: {e}")
        
        if tokenizer is None:
            raise Exception("无法加载tokenizer")
        
        # 手动添加缺失的特殊token属性，解决"has no attribute image_token"的问题
        # 从special_tokens_map.json中读取特殊token
        special_tokens_map_path = os.path.join(MODEL_PATH, "special_tokens_map.json")
        if os.path.exists(special_tokens_map_path):
            import json
            with open(special_tokens_map_path, 'r', encoding='utf-8') as f:
                special_tokens_map = json.load(f)
            
            # 添加缺失的属性
            if "additional_special_tokens" in special_tokens_map:
                special_tokens = special_tokens_map["additional_special_tokens"]
                for token in special_tokens:
                    if token == "<|IMAGE|>" and not hasattr(tokenizer, 'image_token'):
                        tokenizer.image_token = token
                    elif token == "<|AUDIO|>" and not hasattr(tokenizer, 'audio_token'):
                        tokenizer.audio_token = token
                    elif token == "<|VIDEO|>" and not hasattr(tokenizer, 'video_token'):
                        tokenizer.video_token = token
        
        # 如果仍然没有这些属性，手动设置默认值
        if not hasattr(tokenizer, 'image_token'):
            tokenizer.image_token = "<|IMAGE|>"
        if not hasattr(tokenizer, 'audio_token'):
            tokenizer.audio_token = "<|AUDIO|>"
        if not hasattr(tokenizer, 'video_token'):
            tokenizer.video_token = "<|VIDEO|>"
        
        # 加载处理器（使用 fast 版本以避免警告）
        processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("模型加载成功!")
        return model, tokenizer, processor
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def chat_with_model(user_input: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    与模型对话
    """
    # 加载模型（如果尚未加载）
    model, tokenizer, processor = load_model()
    
    if model is None or tokenizer is None or processor is None:
        return "模型加载失败，请检查控制台输出。", history
    
    try:
        # 准备对话历史
        messages = [{"role": "system", "content": "You are Qwen, a large-scale language model developed by Tongyi Lab."}]
        
        # 添加历史对话
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 应用聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成响应
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        
        # 去除输入部分，只保留生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码生成的文本
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 更新历史记录
        history.append((user_input, response))
        return response, history
    except Exception as e:
        error_msg = f"处理过程中出错: {str(e)}"
        print(error_msg)
        return error_msg, history

def reset_history():
    """
    重置对话历史
    """
    return []

# Gradio 界面
with gr.Blocks(title="Qwen2.5-Omni 本地模型") as demo:
    gr.Markdown("# Qwen2.5-Omni 本地模型演示")
    gr.Markdown("这是一个在本地运行的 Qwen2.5-Omni 模型，支持文本对话功能。")
    
    # 存储对话历史
    chat_history = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=3):
            # 聊天机器人界面
            chatbot = gr.Chatbot(
                label="对话",
                bubble_full_width=False,
                height=500
            )
            
            # 用户输入
            user_input = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的问题...",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("提交", variant="primary")
                clear_btn = gr.Button("清除历史")
        
        with gr.Column(scale=1):
            gr.Markdown("## 使用说明")
            gr.Markdown("""
            1. 在文本框中输入您的问题
            2. 点击"提交"按钮或按回车键发送
            3. 等待模型生成回复
            4. 使用"清除历史"按钮重置对话
            
            **注意**: 首次运行时需要加载模型，可能需要一些时间。
            """)
            
            gr.Markdown("## 模型信息")
            gr.Markdown("""
            - 模型: Qwen2.5-Omni
            - 位置: [/data/qwen_omni/](file:///data/qwen_omni/)
            - 类型: 本地运行
            """)
    
    # 事件处理
    submit_btn.click(
        fn=chat_with_model,
        inputs=[user_input, chat_history],
        outputs=[user_input, chat_history]
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=[],
        outputs=[user_input]
    )
    
    user_input.submit(
        fn=chat_with_model,
        inputs=[user_input, chat_history],
        outputs=[user_input, chat_history]
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=[],
        outputs=[user_input]
    )
    
    clear_btn.click(
        fn=reset_history,
        inputs=[],
        outputs=[chat_history]
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)