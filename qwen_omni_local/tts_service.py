#!/usr/bin/env python3
"""
TTS (文本转语音) 服务模块
使用自定义的Apifox API进行文本转语音
"""

import requests
import json
import os
import logging
from typing import Dict, Any, List, Optional
import base64
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class TTSService:
    """TTS服务类"""

    def __init__(self, api_base_url: str = "https://api.guantoufuzhu.com:18000", project_id: str = "ea136102"):
        """
        初始化TTS服务

        Args:
            api_base_url: TTS API基础URL
            project_id: 项目ID
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.project_id = project_id
        self.audio_output_dir = "tts_audio"

        # 创建音频输出目录
        os.makedirs(self.audio_output_dir, exist_ok=True)

        # 构建TTS API端点
        self.tts_endpoint = f"{self.api_base_url}/{self.project_id}/submit-tts"

        logger.info(f"TTS服务初始化完成: {self.tts_endpoint}")

        # 测试连接
        self._test_connection()

    def _test_connection(self):
        """测试TTS API连接"""
        logger.info("🔍 测试TTS API连接...")

        # 尝试多次连接
        for attempt in range(3):
            try:
                logger.info(f"尝试连接TTS API (第{attempt + 1}次)...")
                # 使用简单文本测试连接
                test_response = requests.post(
                    self.tts_endpoint,
                    params={"tts_text": "Hello"},
                    timeout=15  # 增加超时时间
                )

                if test_response.status_code in [200, 201, 202]:
                    logger.info(f"✅ TTS API连接成功: {self.tts_endpoint}")
                    return
                elif 500 <= test_response.status_code < 600:
                    logger.warning(f"⚠️ TTS API服务器错误: {test_response.status_code}")
                else:
                    logger.warning(f"⚠️ TTS API响应异常: {test_response.status_code}")

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"❌ 无法连接到TTS API: {self.tts_endpoint}, 错误: {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ TTS API连接超时 (第{attempt + 1}次)")
            except Exception as e:
                logger.warning(f"❌ TTS API连接错误: {e}")
            
            # 在重试前等待一段时间
            if attempt < 2:  # 不在最后一次尝试后等待
                import time
                time.sleep(2)

        logger.warning("❌ TTS API连接测试失败，将在实际使用时再次尝试连接")

    def text_to_speech(self, text: str, voice: str = "default", speed: float = 1.0, sentence_index: Optional[int] = None) -> Optional[str]:
        """
        将文本转换为语音

        Args:
            text: 要转换的文本
            voice: 语音类型（暂时未使用）
            speed: 语速（暂时未使用）
            sentence_index: 句子索引（用于分句）

        Returns:
            str: 生成的音频文件路径，失败返回None
        """
        try:
            import time
            logger.info(f"🔊 开始TTS转换: {text[:50]}...")

            # 尝试多次请求
            for attempt in range(3):
                try:
                    # 使用你的真实API端点
                    params = {
                        "tts_text": text
                    }

                    headers = {
                        "Content-Type": "application/x-www-form-urlencoded"
                    }

                    # 发送POST请求，增加超时时间
                    response = requests.post(
                        self.tts_endpoint,
                        params=params,
                        headers=headers,
                        timeout=90  # 增加超时时间到90秒，因为TTS可能需要更长时间
                    )

                    logger.info(f"TTS API响应状态: {response.status_code}")

                    if response.status_code in [200, 201, 202]:
                        # 检查响应内容类型
                        content_type = response.headers.get('content-type', '').lower()

                        if 'audio' in content_type or 'octet-stream' in content_type:
                            # 直接返回音频数据
                            audio_data = response.content
                            filename = self._save_audio_file(audio_data, text, sentence_index)
                            logger.info(f"✅ TTS转换成功: {filename}")
                            return filename

                        elif 'json' in content_type:
                            # JSON响应
                            try:
                                result = response.json()
                                logger.info(f"TTS API JSON响应: {result}")

                                if "audio_data" in result:
                                    # Base64编码的音频数据
                                    audio_data = base64.b64decode(result["audio_data"])
                                    filename = self._save_audio_file(audio_data, text, sentence_index)
                                    logger.info(f"✅ TTS转换成功: {filename}")
                                    return filename

                                elif "audio_url" in result:
                                    # 音频URL
                                    audio_url = result["audio_url"]
                                    filename = self._download_audio_file(audio_url, text)
                                    logger.info(f"✅ TTS转换成功: {filename}")
                                    return filename

                                elif "task_id" in result:
                                    # 异步任务，需要轮询结果
                                    logger.info(f"TTS任务已提交，任务ID: {result['task_id']}")
                                    return self._poll_tts_result(result['task_id'], text)

                                else:
                                    logger.warning(f"未知的JSON响应格式: {result}")
                                    return None

                            except json.JSONDecodeError:
                                logger.error("❌ 无法解析JSON响应")
                                return None

                        else:
                            # 尝试作为音频数据处理
                            if len(response.content) > 1000:  # 假设音频文件至少1KB
                                audio_data = response.content
                                filename = self._save_audio_file(audio_data, text, sentence_index)
                                logger.info(f"✅ TTS转换成功: {filename}")
                                return filename
                            else:
                                logger.error(f"❌ 响应内容太小，可能不是音频文件: {len(response.content)} bytes")
                                logger.error(f"响应内容: {response.text[:200]}")
                                return None

                    else:
                        logger.error(f"❌ TTS API请求失败: {response.status_code}")
                        logger.error(f"响应内容: {response.text[:500]}")
                        # 对于5xx服务器错误，进行重试
                        if 500 <= response.status_code < 600 and attempt < 2:
                            logger.warning(f"服务器错误，将在5秒后重试 (第{attempt + 1}次)")
                            time.sleep(5)
                            continue
                        return None

                except requests.exceptions.Timeout:
                    logger.warning(f"⏰ TTS API请求超时 (第{attempt + 1}次)")
                    if attempt < 2:  # 不在最后一次尝试后等待
                        time.sleep(5)  # 等待5秒后重试
                except requests.exceptions.ConnectionError:
                    logger.error("❌ 无法连接到TTS API")
                    if attempt < 2:  # 不在最后一次尝试后等待
                        time.sleep(5)  # 等待5秒后重试
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ TTS API请求异常: {e}")
                    if attempt < 2:  # 不在最后一次尝试后等待
                        time.sleep(5)  # 等待5秒后重试
                except Exception as e:
                    logger.error(f"❌ TTS转换失败: {e}")
                    if attempt < 2:  # 不在最后一次尝试后等待
                        time.sleep(5)  # 等待5秒后重试
                        
            return None  # 所有尝试都失败了
            
        except Exception as e:
            logger.error(f"❌ TTS转换出现未处理的异常: {e}")
            return None

    def _poll_tts_result(self, task_id: str, text: str, max_attempts: int = 60) -> Optional[str]:
        """轮询TTS任务结果"""
        import time

        for attempt in range(max_attempts):
            try:
                # 构建查询URL
                query_url = f"{self.api_base_url}/{self.project_id}/query-tts"
                params = {"task_id": task_id}

                response = requests.get(query_url, params=params, timeout=15)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "completed":
                        if "audio_data" in result:
                            audio_data = base64.b64decode(result["audio_data"])
                            filename = self._save_audio_file(audio_data, text)
                            logger.info(f"✅ 异步TTS任务完成: {filename}")
                            return filename
                        elif "audio_url" in result:
                            filename = self._download_audio_file(result["audio_url"], text)
                            logger.info(f"✅ 异步TTS任务完成: {filename}")
                            return filename

                    elif result.get("status") == "failed":
                        logger.error(f"❌ TTS任务失败: {result.get('error', '未知错误')}")
                        return None

                    elif result.get("status") in ["pending", "processing"]:
                        logger.info(f"⏳ TTS任务处理中... ({attempt+1}/{max_attempts})")
                        time.sleep(5)  # 增加等待时间到5秒
                        continue

                else:
                    logger.warning(f"查询TTS任务状态失败: {response.status_code}")
                    time.sleep(5)  # 等待后重试

            except Exception as e:
                logger.warning(f"查询TTS任务状态异常: {e}")
                time.sleep(5)  # 等待后重试

        logger.error("❌ TTS任务轮询超时")
        return None

    def batch_text_to_speech(self, texts: List[str], voice: str = "default", speed: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量文本转语音
        
        Args:
            texts: 文本列表
            voice: 语音类型
            speed: 语速
            
        Returns:
            List[Dict]: 转换结果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"🔊 批量TTS转换 {i+1}/{len(texts)}")
            
            audio_file = self.text_to_speech(text, voice, speed, i+1)
            
            result = {
                "index": i + 1,
                "text": text,
                "audio_file": audio_file,
                "success": audio_file is not None
            }
            
            results.append(result)
        
        successful_count = len([r for r in results if r['success']])
        logger.info(f"✅ 批量TTS转换完成: {successful_count}/{len(results)} 成功")
        
        if successful_count < len(results):
            logger.warning(f"⚠️ {len(results) - successful_count} 个TTS转换失败")
            
        return results
    
    def convert_evaluation_result(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换评估结果中的文本为语音
        
        Args:
            evaluation_result: 评估结果
            
        Returns:
            Dict: 包含音频文件的评估结果
        """
        try:
            logger.info("🎵 开始转换评估结果为语音...")
            
            result = evaluation_result.copy()
            result["tts_files"] = {}
            
            # 转换完整转录文本
            if "transcription" in evaluation_result:
                transcription = evaluation_result["transcription"]
                logger.info("🔊 转换完整转录文本...")
                
                full_audio = self.text_to_speech(transcription, voice="default", speed=1.0)
                if full_audio:
                    result["tts_files"]["full_transcription"] = {
                        "text": transcription,
                        "audio_file": full_audio,
                        "type": "full_text"
                    }
            
            # 转换分句文本 (修复：使用正确的键sentence_analysis而不是sentences)
            if "sentence_analysis" in evaluation_result and evaluation_result["sentence_analysis"]:
                logger.info("🔊 转换分句文本...")
                
                sentence_texts = [sentence.get("text", "") for sentence in evaluation_result["sentence_analysis"]]
                sentence_audios = self.batch_text_to_speech(sentence_texts, voice="default", speed=1.0)
                
                result["tts_files"]["sentences"] = []
                for i, sentence_data in enumerate(evaluation_result["sentence_analysis"]):
                    if i < len(sentence_audios):
                        audio_result = sentence_audios[i]
                        sentence_with_audio = sentence_data.copy()
                        if audio_result and audio_result["success"]:
                            sentence_with_audio["audio_file"] = audio_result["audio_file"]
                            logger.info(f"✅ 第 {i+1} 个分句音频生成成功: {audio_result['audio_file']}")
                        else:
                            sentence_with_audio["audio_file"] = None
                            logger.warning(f"⚠️ 第 {i+1} 个分句音频生成失败")
                    else:
                        sentence_with_audio = sentence_data.copy()
                        sentence_with_audio["audio_file"] = None
                        logger.warning(f"⚠️ 第 {i+1} 个分句没有对应的音频生成结果")
                    
                    result["tts_files"]["sentences"].append(sentence_with_audio)
            
            # 添加TTS元数据
            result["tts_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "total_files": len([f for f in result["tts_files"].values() if f]),
                "api_endpoint": self.api_base_url,
                "project_id": self.project_id
            }
            
            logger.info("✅ 评估结果语音转换完成")
            return result
            
        except Exception as e:
            logger.error(f"❌ 评估结果语音转换失败: {e}")
            result = evaluation_result.copy()
            result["tts_error"] = str(e)
            return result
    
    def _save_audio_file(self, audio_data: bytes, text: str, sentence_index: Optional[int] = None) -> str:
        """保存音频文件"""
        # 生成文件名
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据是否是分句生成不同的文件名
        if sentence_index is not None:
            filename = f"sentence_{sentence_index}_{text_hash}.wav"
        else:
            filename = f"tts_{timestamp}_{text_hash}.wav"
            
        filepath = os.path.join(self.audio_output_dir, filename)
        
        # 保存文件
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        return filepath
    
    def _download_audio_file(self, audio_url: str, text: str) -> str:
        """下载音频文件"""
        try:
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            # 生成文件名
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{timestamp}_{text_hash}.wav"
            filepath = os.path.join(self.audio_output_dir, filename)
            
            # 保存文件
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 下载音频文件失败: {e}")
            raise
    
    def get_available_voices(self) -> List[str]:
        """获取可用的语音类型"""
        try:
            voices_url = f"{self.api_base_url}/api/voices"
            response = requests.get(voices_url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("voices", ["default"])
            else:
                logger.warning("无法获取语音类型列表，使用默认值")
                return ["default", "male", "female"]
                
        except Exception as e:
            logger.error(f"获取语音类型失败: {e}")
            return ["default", "male", "female"]
    
    def cleanup_old_files(self, max_files: int = 50):
        """清理旧的音频文件"""
        try:
            if not os.path.exists(self.audio_output_dir):
                return
            
            files = [f for f in os.listdir(self.audio_output_dir) if f.endswith('.wav')]
            files.sort(key=lambda x: os.path.getctime(os.path.join(self.audio_output_dir, x)))
            
            if len(files) > max_files:
                files_to_remove = files[:-max_files]
                for file in files_to_remove:
                    os.remove(os.path.join(self.audio_output_dir, file))
                
                logger.info(f"🗑️ 清理了 {len(files_to_remove)} 个旧音频文件")
                
        except Exception as e:
            logger.error(f"清理音频文件失败: {e}")

def test_tts_service():
    """测试TTS服务"""
    try:
        tts = TTSService()
        
        # 测试单个文本转换
        test_text = "Hello, this is a test of the text to speech service."
        audio_file = tts.text_to_speech(test_text)
        
        if audio_file:
            print(f"✅ TTS测试成功: {audio_file}")
        else:
            print("❌ TTS测试失败")
        
        # 测试批量转换
        test_texts = [
            "Hello, my name is John.",
            "I am learning English.",
            "This is a test sentence."
        ]
        
        batch_results = tts.batch_text_to_speech(test_texts)
        successful = len([r for r in batch_results if r['success']])
        print(f"✅ 批量TTS测试: {successful}/{len(test_texts)} 成功")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS服务测试失败: {e}")
        return False

if __name__ == "__main__":
    test_tts_service()
