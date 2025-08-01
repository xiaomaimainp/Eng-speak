#!/usr/bin/env python3
"""
配置管理器 - 处理YAML配置文件
"""

import yaml
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "prompts.yml"):
        """
        初始化配置管理器
        
        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ 配置文件加载成功: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.config.get('system_prompt', '')
    
    def get_evaluation_prompt(self, prompt_type: str = 'detailed_evaluation') -> Dict[str, str]:
        """
        获取评估提示词
        
        Args:
            prompt_type: 提示词类型
            
        Returns:
            Dict[str, str]: 包含name, description, prompt的字典
        """
        evaluation_prompts = self.config.get('evaluation_prompts', {})
        
        if prompt_type not in evaluation_prompts:
            available_types = list(evaluation_prompts.keys())
            raise ValueError(f"未知的提示词类型: {prompt_type}. 可用类型: {available_types}")
        
        return evaluation_prompts[prompt_type]
    
    def list_available_prompts(self) -> List[Dict[str, str]]:
        """列出所有可用的提示词"""
        evaluation_prompts = self.config.get('evaluation_prompts', {})
        result = []
        
        for key, value in evaluation_prompts.items():
            result.append({
                'type': key,
                'name': value.get('name', key),
                'description': value.get('description', '无描述')
            })
        
        return result
    
    def get_output_format(self) -> Dict[str, Any]:
        """获取输出格式配置"""
        return self.config.get('output_format', {})
    
    def get_sentence_segmentation_config(self) -> Dict[str, Any]:
        """获取分句配置"""
        return self.config.get('sentence_segmentation', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config.get('evaluation_config', {})
    
    def build_full_prompt(self, prompt_type: str, transcription: str, sentence_analysis: List[Dict] = None) -> str:
        """
        构建完整的评估提示词
        
        Args:
            prompt_type: 提示词类型
            transcription: 转录文本
            sentence_analysis: 分句分析结果
            
        Returns:
            str: 完整的提示词
        """
        system_prompt = self.get_system_prompt()
        evaluation_prompt_data = self.get_evaluation_prompt(prompt_type)
        evaluation_prompt = evaluation_prompt_data['prompt']
        
        # 构建分句分析部分
        sentence_info = ""
        if sentence_analysis:
            sentence_info = "\n**分句分析结果：**\n"
            for i, sentence_data in enumerate(sentence_analysis, 1):
                sentence_info += f"{i}. \"{sentence_data['text']}\" (长度: {sentence_data['word_count']}词)\n"
        
        # 构建完整提示词
        full_prompt = f"""{system_prompt}

{evaluation_prompt}

**需要评估的英语口语内容：**
原文: "{transcription}"
{sentence_info}

**输出要求：**
请严格按照JSON格式输出评估结果，包含以下字段：
- overall_score: 总体评分 (0-100)
- transcription: 完整转录文本
- sentence_analysis: 分句分析结果
- evaluation: 各维度详细评分和反馈
- detailed_analysis: 深入分析（优势、问题、错误示例）
- suggestions: 具体改进建议
- metadata: 元数据信息

请确保评估具体、详细、实用，提供可操作的改进建议。"""

        return full_prompt
    
    def validate_config(self) -> bool:
        """验证配置文件完整性"""
        required_sections = [
            'system_prompt',
            'evaluation_prompts',
            'output_format',
            'sentence_segmentation',
            'evaluation_config'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"❌ 配置文件缺少必要部分: {missing_sections}")
            return False
        
        # 验证评估提示词
        evaluation_prompts = self.config.get('evaluation_prompts', {})
        if not evaluation_prompts:
            logger.error("❌ 没有配置评估提示词")
            return False
        
        for prompt_type, prompt_data in evaluation_prompts.items():
            if 'prompt' not in prompt_data:
                logger.error(f"❌ 提示词 {prompt_type} 缺少 prompt 字段")
                return False
        
        logger.info("✅ 配置文件验证通过")
        return True
    
    def reload_config(self):
        """重新加载配置文件"""
        self.config = self._load_config()
        logger.info("🔄 配置文件已重新加载")
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """获取评分权重"""
        eval_config = self.get_evaluation_config()
        return eval_config.get('scoring', {}).get('weights', {
            'pronunciation': 0.25,
            'fluency': 0.25,
            'grammar': 0.20,
            'vocabulary': 0.20,
            'content': 0.10
        })
    
    def get_grade_boundaries(self) -> Dict[str, List[int]]:
        """获取评分等级边界"""
        eval_config = self.get_evaluation_config()
        return eval_config.get('scoring', {}).get('grade_boundaries', {
            'excellent': [90, 100],
            'good': [80, 89],
            'fair': [70, 79],
            'pass': [60, 69],
            'fail': [0, 59]
        })
    
    def get_grade_for_score(self, score: int) -> str:
        """根据分数获取等级"""
        boundaries = self.get_grade_boundaries()
        
        for grade, (min_score, max_score) in boundaries.items():
            if min_score <= score <= max_score:
                return grade
        
        return 'unknown'

def test_config_manager():
    """测试配置管理器"""
    try:
        config_manager = ConfigManager()
        
        # 验证配置
        if not config_manager.validate_config():
            print("❌ 配置验证失败")
            return False
        
        # 测试获取提示词
        prompts = config_manager.list_available_prompts()
        print(f"✅ 可用提示词数量: {len(prompts)}")
        for prompt in prompts:
            print(f"  - {prompt['type']}: {prompt['name']}")
        
        # 测试构建完整提示词
        test_transcription = "Hello, my name is John. I am very excited to be here today."
        full_prompt = config_manager.build_full_prompt('detailed_evaluation', test_transcription)
        print(f"✅ 完整提示词长度: {len(full_prompt)} 字符")
        
        # 测试评分等级
        test_scores = [95, 85, 75, 65, 55]
        for score in test_scores:
            grade = config_manager.get_grade_for_score(score)
            print(f"  分数 {score} -> 等级 {grade}")
        
        print("✅ 配置管理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False

if __name__ == "__main__":
    test_config_manager()
