import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import soundfile as sf
import numpy as np
import sys

# 导入Qwen2.5 Omni模型，但尝试以文本模式使用
try:
    from transformers import Qwen2_5OmniForConditionalGeneration # type: ignore
    QWEN_OMNI_AVAILABLE = True
    print("成功导入Qwen2_5OmniForConditionalGeneration")
except ImportError:
    QWEN_OMNI_AVAILABLE = False
    print("Qwen2_5OmniForConditionalGeneration不可用，将使用AutoModelForCausalLM")

# 模型路径
MODEL_PATH = "/data/qwen_omni/"

class CustomQwen2Tokenizer:
    """
    自定义Tokenizer类，用于解决缺少image_token等属性的问题
    """
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # 添加缺失的属性
        self.image_token = "<|IMAGE|>"
        self.audio_token = "<|AUDIO|>"
        self.video_token = "<|VIDEO|>"

        # 将原始tokenizer的所有属性复制到当前实例
        for attr in dir(tokenizer):
            if not attr.startswith('__') and not hasattr(self, attr):
                setattr(self, attr, getattr(tokenizer, attr))

    def __call__(self, *args, **kwargs):
        # 实现__call__方法，将调用转发给原始tokenizer
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        # 如果当前类没有该属性，则从原始tokenizer获取
        return getattr(self._tokenizer, name)

def load_model():
    """
    加载 Qwen2.5-Omni 模型
    """
    print("正在加载模型...")
    try:
        # 先加载原始tokenizer
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 使用自定义tokenizer包装原始tokenizer，确保具有所有必要属性
        custom_tokenizer = CustomQwen2Tokenizer(tokenizer)

        # 创建一个简化的processor类，避免初始化问题
        class SimpleProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                # 添加必要的属性
                self.image_token = "<|IMAGE|>"
                self.audio_token = "<|AUDIO|>"
                self.video_token = "<|VIDEO|>"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                """
                简化的聊天模板应用函数
                """
                # tokenize参数在这个简化版本中不使用，但保留以兼容接口
                _ = tokenize  # 避免未使用参数警告

                if not messages:
                    return ""

                # 构建聊天格式的文本
                formatted_text = ""
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")

                    if role == "system":
                        formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

                if add_generation_prompt:
                    formatted_text += "<|im_start|>assistant\n"

                return formatted_text

        # 使用简化的processor
        processor = SimpleProcessor(custom_tokenizer)

        # 加载模型
        print("正在加载模型...")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 尝试使用GPU，如果失败则回退到CPU
        try:
            if torch.cuda.is_available():
                print("尝试使用GPU设备")
                device_map = None  # 不使用auto，手动管理设备
                torch_dtype = torch.float16

                # 根据可用性选择模型类
                if QWEN_OMNI_AVAILABLE:
                    print("使用Qwen2_5OmniForConditionalGeneration")
                    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    # 手动移动到GPU
                    model = model.to('cuda')
                else:
                    print("使用AutoModelForCausalLM")
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    model = model.to('cuda')
                print("成功使用GPU")
            else:
                raise Exception("CUDA不可用")

        except Exception as e:
            print(f"GPU加载失败: {e}")
            print("回退到CPU设备")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            device_map = "cpu"
            torch_dtype = torch.float32

            # 根据可用性选择模型类
            if QWEN_OMNI_AVAILABLE:
                print("使用Qwen2_5OmniForConditionalGeneration (CPU)")
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                print("使用AutoModelForCausalLM (CPU)")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

        print("模型加载成功!")
        return model, custom_tokenizer, processor
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def text_to_text_inference(model, tokenizer, processor, text_input):
    """
    文本到文本推理
    """
    try:
        print("开始处理输入...")
        # 准备输入
        messages = [
            {"role": "system", "content": "You are Qwen, a large-scale language model developed by Tongyi Lab."},
            {"role": "user", "content": text_input}
        ]

        # 应用聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 设置pad_token_id以避免警告
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print("编码输入...")
        # 编码输入
        model_inputs = tokenizer([text], return_tensors="pt", padding=True)

        # 确保输入在正确的设备上
        print("将输入移至正确设备...")
        # 获取模型实际所在的设备
        try:
            # 获取模型参数的设备
            model_device = next(model.parameters()).device
            print(f"模型设备: {model_device}")
            device = model_device
        except Exception as e:
            print(f"获取模型设备失败: {e}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {device}")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # 生成响应
        print("开始生成响应...")

        # 对于Qwen2.5 Omni模型，尝试禁用音频功能，只进行文本生成
        generation_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs.get("attention_mask"),
            "max_new_tokens": 128,  # 减少生成长度以加快速度
            "do_sample": False,  # 使用贪婪解码以提高稳定性
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            # 添加CPU优化参数
            "num_beams": 1,  # 禁用beam search
            "early_stopping": False,  # 禁用early stopping以避免兼容性问题
        }

        # 如果是Qwen2.5 Omni模型，添加特殊参数来禁用音频功能
        original_audio_setting = None
        original_talker_setting = None

        if QWEN_OMNI_AVAILABLE and hasattr(model, 'config'):
            if hasattr(model.config, 'enable_audio_output'):
                # 尝试临时禁用音频输出
                original_audio_setting = model.config.enable_audio_output
                model.config.enable_audio_output = False
                print("已禁用音频输出功能")

            if hasattr(model.config, 'enable_talker'):
                original_talker_setting = model.config.enable_talker
                model.config.enable_talker = False
                print("已禁用talker功能")

        try:
            print("正在生成文本，请耐心等待（CPU推理较慢）...")
            with torch.no_grad():
                generated_ids = model.generate(**generation_kwargs)
            print("文本生成完成！")
        except Exception as e:
            print(f"生成失败，尝试更简单的参数: {e}")
            # 如果失败，尝试最简单的生成参数
            print("使用简化参数重新生成...")
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs["input_ids"],
                    max_new_tokens=64,  # 进一步减少长度
                    pad_token_id=tokenizer.pad_token_id
                )
            print("简化生成完成！")
        finally:
            # 恢复原始设置
            if QWEN_OMNI_AVAILABLE and hasattr(model, 'config'):
                if hasattr(model.config, 'enable_audio_output') and original_audio_setting is not None:
                    model.config.enable_audio_output = original_audio_setting
                if hasattr(model.config, 'enable_talker') and original_talker_setting is not None:
                    model.config.enable_talker = original_talker_setting

        print("处理生成结果...")
        try:
            print(f"原始输入长度: {model_inputs['input_ids'].shape}")
            print(f"生成结果类型: {type(generated_ids)}")

            # 处理Qwen2.5 Omni模型的特殊输出格式
            if isinstance(generated_ids, tuple):
                print("检测到tuple输出，提取文本部分...")
                # Qwen2.5 Omni可能返回(text_ids, audio_ids)的tuple
                text_ids = generated_ids[0]  # 假设第一个是文本ID
                print(f"文本ID形状: {text_ids.shape}")
            else:
                text_ids = generated_ids
                print(f"生成结果形状: {text_ids.shape}")

            # 去除输入部分，只保留生成的部分
            input_length = model_inputs["input_ids"].shape[1]
            new_tokens = text_ids[:, input_length:]

            print(f"新生成的token数量: {new_tokens.shape[1]}")

            # 解码生成的文本
            response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            print("生成完成!")
            print(f"模型回复: {response}")
            return response

        except Exception as decode_error:
            print(f"解码过程中出错: {decode_error}")
            import traceback
            traceback.print_exc()

            # 尝试直接解码整个输出
            try:
                if isinstance(generated_ids, tuple):
                    full_response = tokenizer.decode(generated_ids[0][0], skip_special_tokens=True)
                else:
                    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"完整输出: {full_response}")
                return full_response
            except Exception as full_decode_error:
                print(f"完整解码也失败: {full_decode_error}")
                return None
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 加载模型
    model, tokenizer, processor = load_model()
    
    if model is None:
        print("无法加载模型，程序退出。")
        return
    
    # 如果提供了命令行参数，则使用它作为输入
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
    else:
        # 否则提示用户输入
        text_input = input("请输入您的问题: ")
    
    print(f"\n输入: {text_input}")
    
    # 执行推理
    response = text_to_text_inference(model, tokenizer, processor, text_input)
    
    if response:
        print(f"\n模型响应:\n{response}")
    else:
        print("推理失败。")

if __name__ == "__main__":
    main()