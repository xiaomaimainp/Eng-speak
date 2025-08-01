#!/usr/bin/env python3
"""
快速测试Qwen2.5 Omni模型的简化版本
"""
import torch
from transformers import AutoTokenizer
import sys

# 导入Qwen2.5 Omni模型
try:
    from transformers import Qwen2_5OmniForConditionalGeneration  # type: ignore
    QWEN_OMNI_AVAILABLE = True
    print("✅ 成功导入Qwen2_5OmniForConditionalGeneration")
except ImportError as e:
    QWEN_OMNI_AVAILABLE = False
    print(f"❌ Qwen2_5OmniForConditionalGeneration不可用: {e}")
    sys.exit(1)

# 模型路径
MODEL_PATH = "/data/qwen_omni/"

def quick_test():
    """快速测试模型功能"""
    print("🚀 开始快速测试...")
    
    # 加载tokenizer
    print("📝 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 设置pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型（仅CPU，避免GPU内存问题）
    print("🤖 加载模型到CPU...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 禁用音频功能
    if hasattr(model.config, 'enable_audio_output'):
        model.config.enable_audio_output = False
        print("🔇 已禁用音频输出")
    
    if hasattr(model.config, 'enable_talker'):
        model.config.enable_talker = False
        print("🔇 已禁用talker功能")
    
    print("✅ 模型加载完成!")
    
    # 准备输入
    text_input = "Hello, how are you?"
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
    
    print(f"💬 输入: {text_input}")
    
    # 简单的文本格式化
    formatted_text = f"<|im_start|>user\n{text_input}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码
    print("🔤 编码输入...")
    inputs = tokenizer(formatted_text, return_tensors="pt")
    
    # 生成（使用最简单的参数）
    print("⚡ 开始生成（使用最简参数）...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,  # 很短的输出
                do_sample=False,    # 贪婪解码
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print("🎉 生成成功!")
        print(f"🤖 模型回复: {generated_text}")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
