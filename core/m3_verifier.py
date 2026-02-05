# -*- coding: utf-8 -*-
"""
M3: Local VLM Semantic Verification Module
使用 Qwen3-VL-8B 进行图像对的语义验证，判断是否为同一地点。

Model: Qwen/Qwen2-VL-7B-Instruct 或 Qwen3-VL-8B-Instruct
"""

import torch
import cv2
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


class LocalM3Verifier:
    """本地 VLM 语义验证器 - 使用 Qwen3-VL"""

    # ============ Prompt 选项 ============
    PROMPTS = {
        # 选项 1: 简洁直接版 (推荐首选)
        "simple": {
            "system": "You are a visual location matching expert.",
            "user": """Look at these two images carefully.

Image 1 is the query image.
Image 2 is a candidate from the database.

Question: Are these two images taken at the SAME physical location? 
They should show the same scene, building, landmark, or spot from similar or different angles.

Answer with only 'Yes' or 'No'."""
        },
        
        # 选项 2: 景区讲解专用版
        "scenic_guide": {
            "system": "You are a scenic area guide assistant that helps match visitor photos to known locations.",
            "user": """I need to identify if a visitor's photo matches a known scenic spot.

Image 1: Visitor's photo (query)
Image 2: Reference photo from our scenic area database

Task: Determine if both images show the SAME location/spot in the scenic area.
Consider: Same building, same statue, same garden area, same architectural feature, etc.
Ignore: Different weather, lighting, people, or camera angles.

Are these two images from the same location? Answer 'Yes' or 'No' only."""
        },
        
        # 选项 3: 严格匹配版 (减少误报)
        "strict": {
            "system": "You are a precise visual localization system.",
            "user": """Compare these two images for location matching.

Image 1: Query image
Image 2: Database candidate

Criteria for a MATCH (answer 'Yes'):
- Both images must show the SAME specific location
- There should be recognizable shared visual elements (buildings, landmarks, distinctive features)
- Minor differences in angle, lighting, or time are acceptable

Criteria for NO MATCH (answer 'No'):
- Different locations, even if visually similar
- No clear shared distinctive features
- Only generic similarities (e.g., both are gardens, but different gardens)

Based on these criteria, do these images show the same location? Answer 'Yes' or 'No'."""
        },
        
        # 选项 4: 中文版
        "chinese": {
            "system": "你是一个景区图像匹配专家。",
            "user": """请仔细观察这两张图片。

图片1：游客拍摄的照片（查询图）
图片2：景区数据库中的参考照片

任务：判断这两张图片是否拍摄于同一个地点/景点。
判断依据：相同的建筑、雕塑、园林景观、标志性特征等。
忽略因素：光线、天气、人物、拍摄角度的差异。

这两张图片是否来自同一地点？请只回答"是"或"否"。"""
        },
        
        # 选项 5: 详细分析版 (会输出更多信息)
        "detailed": {
            "system": "You are an expert in visual place recognition for scenic areas and tourist attractions.",
            "user": """Analyze these two images for location matching.

Image 1: Query photo from a visitor
Image 2: Reference photo from the scenic area database

Please determine if both images were taken at the SAME physical location within the scenic area.

Focus on:
- Architectural features (buildings, structures, decorations)
- Natural landmarks (trees, rocks, water features)
- Signage or distinctive markers
- Spatial layout and arrangement

Ignore variations in:
- Lighting conditions
- Weather
- Presence of people
- Camera angle or zoom level

Final answer: Are these images from the same location? Reply with 'Yes' or 'No'."""
        }
    }

    # 默认模型路径
    DEFAULT_MODEL_PATH = 'Qwen/Qwen2-VL-7B-Instruct'

    def __init__(
        self, 
        model_path: str = None,
        device: str = "cuda",
        prompt_style: str = "simple",
        max_dim: int = 1024
    ):
        """初始化本地验证器
        
        Args:
            model_path: Qwen-VL 模型路径，可以是:
                - HuggingFace 模型名: 'Qwen/Qwen2-VL-7B-Instruct'
                - 本地路径: '/path/to/model'
            device: 计算设备
            prompt_style: prompt 风格，可选 "simple", "scenic_guide", "strict", "chinese", "detailed"
            max_dim: 图片最大边长
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device
        self.max_dim = max_dim
        
        # 设置 prompt
        if prompt_style not in self.PROMPTS:
            print(f"Warning: Unknown prompt style '{prompt_style}', using 'simple'")
            prompt_style = "simple"
        self.prompt_style = prompt_style
        self.system_prompt = self.PROMPTS[prompt_style]["system"]
        self.user_prompt_template = self.PROMPTS[prompt_style]["user"]
        
        # 模型相关
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """加载 Qwen-VL 模型"""
        print(f"Loading Qwen-VL model from {self.model_path}...")
        
        try:
            # 尝试加载 Qwen3-VL
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        except:
            # 回退到 Qwen2-VL
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        print(f"✅ Loaded Qwen-VL model (prompt style: {self.prompt_style})")

    def resize_image(self, image_path: str) -> Image.Image:
        """缩放图片到最大边 max_dim"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = img.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def verify_pair(
        self, 
        query_image_path: str, 
        candidate_image_path: str
    ) -> Tuple[bool, str, float]:
        """验证图像对是否匹配
        
        Args:
            query_image_path: 查询图片路径
            candidate_image_path: 候选图片路径
            
        Returns:
            Tuple[bool, str, float]: (是否匹配, 完整响应, 置信度)
        """
        from qwen_vl_utils import process_vision_info
        
        # 预处理图片
        query_img = self.resize_image(query_image_path)
        candidate_img = self.resize_image(candidate_image_path)
        
        # 保存临时文件供模型读取
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
            query_img.save(f1.name, 'JPEG', quality=95)
            query_temp = f1.name
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f2:
            candidate_img.save(f2.name, 'JPEG', quality=95)
            candidate_temp = f2.name
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": query_temp},
                        {"type": "image", "image": candidate_temp},
                        {"type": "text", "text": self.user_prompt_template}
                    ]
                }
            ]
            
            # 处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成响应
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    do_sample=False  # 确定性输出
                )
            
            # 解码响应
            full_response = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            
            # 提取最后的回答部分
            response_text = full_response.lower()
            
            # 判断结果
            if self.prompt_style == "chinese":
                is_match = "是" in full_response and "否" not in full_response
            else:
                # 检查最后出现的 yes/no
                yes_pos = response_text.rfind("yes")
                no_pos = response_text.rfind("no")
                
                if yes_pos > no_pos:
                    is_match = True
                elif no_pos > yes_pos:
                    is_match = False
                else:
                    # 都没找到，默认 False
                    is_match = False
            
            # 简单置信度 (基于响应长度，越短越确定)
            confidence = 1.0 if len(full_response.split()[-1]) <= 3 else 0.8
            
            return is_match, full_response, confidence
            
        finally:
            # 清理临时文件
            import os
            os.unlink(query_temp)
            os.unlink(candidate_temp)

    def verify_candidate(
        self, 
        query_image_path: str, 
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证单个候选
        
        Args:
            query_image_path: 查询图片路径
            candidate: 候选信息字典，需包含 'path' 字段
            
        Returns:
            Dict: 包含验证结果的候选信息
        """
        is_match, response, confidence = self.verify_pair(
            query_image_path, 
            str(candidate['path'])
        )
        
        return {
            **candidate,
            'vlm_verified': is_match,
            'vlm_response': response,
            'vlm_confidence': confidence
        }


def get_available_prompts() -> Dict[str, str]:
    """获取所有可用的 prompt 选项及其描述"""
    return {
        "simple": "简洁直接版 - 最基础的是/否判断 (推荐首选)",
        "scenic_guide": "景区讲解专用版 - 针对景区场景优化",
        "strict": "严格匹配版 - 减少误报，更保守的判断",
        "chinese": "中文版 - 使用中文 prompt",
        "detailed": "详细分析版 - 会输出更多分析信息"
    }
