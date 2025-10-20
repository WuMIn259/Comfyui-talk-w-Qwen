import torch
from PIL import Image
import numpy as np
import os
from .qwen3_vl_manager import model_manager

# 首先更新模型管理器以使用正确的模型类
class Qwen3VLModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None

    def load_model(self, model_path):
        if os.path.isdir(model_path):
            print(f"✅ 检测到本地模型路径：{model_path}，优先加载本地文件")
        else:
            print(f"⚠️ 本地路径不存在，将尝试远程下载：{model_path}")

        if self.current_model_path == model_path and self.model is not None:
            return self.model, self.processor

        # 使用正确的模型类和处理器
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_path)
        )
        
        # 加载模型 - 使用官方推荐的方式
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_path)
        )
        
        self.model.eval()
        self.current_model_path = model_path
        print(f"✅ 模型加载完成：{model_path}")
        
        return self.model, self.processor

# 更新单例模式管理模型
model_manager = Qwen3VLModelManager()

# 纯文本对话节点
class Qwen3VL_TextChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen3-VL-4B-Instruct"}),
                "prompt": ("STRING", {"multiline": True, "default": "你好，介绍一下你自己"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "show_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "history": ("LIST", {"default": []}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LIST")
    RETURN_NAMES = ("response", "thinking_process", "updated_history")
    FUNCTION = "chat"
    CATEGORY = "Qwen3-VL"

    def chat(self, model_path, prompt, max_new_tokens, show_thinking=True, history=None):
        """
        纯文本聊天方法 - 基于官方示例修复
        """
        history = history or []
        model, processor = model_manager.load_model(model_path)
        
        print("🚀 开始生成文本回答...")
        
        # 构建提示词
        if show_thinking:
            thinking_prompt = f"""请先思考再回答：
问题：{prompt}

请按以下格式回答：
思考：<你的思考过程>
回答：<你的最终回答>
"""
        else:
            thinking_prompt = prompt
        
        # 修改为（正确代码）
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": thinking_prompt
                    }
                ]
            }
        ]

        
        try:
            # 使用官方示例的方法
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens
                )
            
            # 官方推荐的解码方式
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            full_response = output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return ("生成失败", f"生成错误: {str(e)}", history)
        
        # 尝试分离思考过程和最终回答
        thinking_process = ""
        final_answer = full_response
        
        if show_thinking and "思考：" in full_response and "回答：" in full_response:
            # 分离思考过程和最终回答
            thinking_start = full_response.find("思考：")
            answer_start = full_response.find("回答：")
            
            if thinking_start < answer_start:
                thinking_process = full_response[thinking_start + 3:answer_start].strip()
                final_answer = full_response[answer_start + 3:].strip()
                
                print(f"🤔 思考过程：{thinking_process}")
                print(f"💡 最终回答：{final_answer}")
        elif show_thinking:
            thinking_process = "模型未按指定格式显示思考过程"
            print(f"💡 回答：{final_answer}")
        else:
            thinking_process = "思考过程已隐藏"
            print(f"💡 回答：{final_answer}")
        
        updated_history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer}
        ]
        
        print("✅ 文本生成完成！")
        
        return (final_answer, thinking_process, updated_history)

# 多模态对话节点
class Qwen3VL_MultimodalChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen3-VL-4B-Instruct"}),
                "prompt": ("STRING", {"multiline": True, "default": "详细描述这张图片的内容"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "show_thinking": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "history": ("LIST", {"default": []}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LIST")
    RETURN_NAMES = ("response", "thinking_process", "updated_history")
    FUNCTION = "chat"
    CATEGORY = "Qwen3-VL"

    def chat(self, model_path, prompt, max_new_tokens, show_thinking=True, image=None, history=None):
        """
        多模态聊天方法 - 基于官方示例完全重写
        """
        history = history or []
        model, processor = model_manager.load_model(model_path)
        
        # 处理图像
        processed_image = None
        if image is not None:
            try:
                # 正确处理批次图像
                if len(image.shape) == 4:  # [batch, height, width, channels]
                    img_tensor = image[0]  # 取第一张图片
                else:
                    img_tensor = image
                
                # 确保数值范围正确并转换为PIL图像
                if img_tensor.max() <= 1.0:
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    img_np = img_tensor.cpu().numpy().astype(np.uint8)
                
                # 处理通道维度
                if len(img_np.shape) == 3 and img_np.shape[-1] == 3:
                    processed_image = Image.fromarray(img_np)
                else:
                    # 如果是单通道或其他格式，转换为RGB
                    processed_image = Image.fromarray(img_np.squeeze()).convert('RGB')
                
                print(f"🖼️ 成功处理图像输入，尺寸: {processed_image.size}")
            except Exception as e:
                print(f"⚠️ 处理图像时出错: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
                return ("图像处理失败", f"图像处理错误: {str(e)}", history)
        else:
            return ("错误", "未提供图像输入", history)
        
        print("🚀 开始生成多模态回答...")
        
        # 构建多模态消息 - 完全按照官方格式
        if show_thinking:
            thinking_prompt = f"""请先观察图片并思考，然后回答：
问题：{prompt}

请按以下格式回答：
思考：<你的观察和思考过程>
回答：<你的最终回答>
"""
        else:
            thinking_prompt = prompt
        
        try:
            # 使用官方示例的多模态消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": processed_image,  # 直接使用PIL图像对象
                        },
                        {
                            "type": "text", 
                            "text": thinking_prompt
                        }
                    ]
                }
            ]
            
            # 使用官方示例的预处理方法
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            print("✅ 成功预处理多模态输入")
            
        except Exception as e:
            print(f"⚠️ 消息预处理失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return ("消息预处理失败", f"预处理错误: {str(e)}", history)
        
        # 生成回答
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens
                )
            
            # 官方推荐的解码方式 - 跳过输入部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            full_response = output_text[0] if output_text else ""
            
            print("✅ 多模态生成成功！")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return ("生成失败", f"生成错误: {str(e)}", history)
        
        # 尝试分离思考过程和最终回答
        thinking_process = ""
        final_answer = full_response
        
        if show_thinking and "思考：" in full_response and "回答：" in full_response:
            # 分离思考过程和最终回答
            thinking_start = full_response.find("思考：")
            answer_start = full_response.find("回答：")
            
            if thinking_start < answer_start:
                thinking_process = full_response[thinking_start + 3:answer_start].strip()
                final_answer = full_response[answer_start + 3:].strip()
                
                print(f"🤔 思考过程：{thinking_process}")
                print(f"💡 最终回答：{final_answer}")
        elif show_thinking:
            thinking_process = "模型未按指定格式显示思考过程"
            print(f"💡 回答：{final_answer}")
        else:
            thinking_process = "思考过程已隐藏"
            print(f"💡 回答：{final_answer}")
        
        updated_history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer}
        ]
        
        print("✅ 多模态对话完成！")
        
        return (final_answer, thinking_process, updated_history)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_TextChat": Qwen3VL_TextChat,
    "Qwen3VL_MultimodalChat": Qwen3VL_MultimodalChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_TextChat": "Qwen3-VL 文本对话",
    "Qwen3VL_MultimodalChat": "Qwen3-VL 多模态对话"
}
