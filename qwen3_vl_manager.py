import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
import os

class Qwen3VLModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None

    def load_model(self, model_path):
        if os.path.isdir(model_path):
            print(f"✅ 检测到本地模型路径：{model_path}，优先加载本地文件")
        else:
            print(f"⚠️ 本地路径不存在，将尝试远程下载：{model_path}")

        if self.current_model_path == model_path and self.model is not None:
            return self.model, self.tokenizer

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_path)
        )
        
        # 完全禁用量化，避免bitsandbytes兼容性问题
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=os.path.isdir(model_path)
        )
        
        self.model.eval()
        self.current_model_path = model_path
        print(f"✅ 模型加载完成：{model_path}")
        
        return self.model, self.tokenizer

# 单例模式管理模型
model_manager = Qwen3VLModelManager()
