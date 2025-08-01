#!/usr/bin/env python3
"""
Qwen2.5 Omni 口语评估 Web 应用
支持音频文件上传和在线评估
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import subprocess
import tempfile
from tts_service import TTSService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# 初始化TTS服务
tts_service = TTSService(
    api_base_url="https://api.guantoufuzhu.com:18000",
    project_id="ea136102"
)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 Qwen2.5 Omni 口语评估系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white; 
            padding: 30px; 
            text-align: center; 
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .content { padding: 40px; }
        .upload-area { 
            border: 3px dashed #4facfe; 
            border-radius: 15px; 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0;
            background: #f8f9ff;
            transition: all 0.3s ease;
        }
        .upload-area:hover { 
            border-color: #667eea; 
            background: #f0f2ff;
        }
        .upload-area.dragover { 
            border-color: #00f2fe; 
            background: #e6f7ff; 
        }
        .file-input { display: none; }
        .upload-btn { 
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 25px; 
            font-size: 1.1em; 
            cursor: pointer; 
            transition: transform 0.2s;
        }
        .upload-btn:hover { transform: translateY(-2px); }
        .form-group { margin: 20px 0; }
        .form-group label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 600; 
            color: #333;
        }
        .form-group select, .form-group input { 
            width: 100%; 
            padding: 12px; 
            border: 2px solid #e1e5e9; 
            border-radius: 8px; 
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group select:focus, .form-group input:focus { 
            outline: none; 
            border-color: #4facfe; 
        }
        .evaluate-btn { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            padding: 12px 30px; 
            border-radius: 25px; 
            font-size: 1em; 
            cursor: pointer; 
            width: 100%;
            margin: 20px 0;
            transition: transform 0.2s;
        }
        .evaluate-btn:hover { transform: translateY(-2px); }
        .evaluate-btn:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
            transform: none;
        }
        .result-area { 
            margin-top: 30px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 15px; 
            display: none;
        }
        .loading { 
            text-align: center; 
            padding: 40px; 
            color: #667eea;
        }
        .spinner { 
            border: 4px solid #f3f3f3; 
            border-top: 4px solid #667eea; 
            border-radius: 50%; 
            width: 40px; 
            height: 40px; 
            animation: spin 1s linear infinite; 
            margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .score-card { 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 15px 0; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .score-header { 
            font-size: 1.5em; 
            color: #333; 
            margin-bottom: 15px; 
            text-align: center;
        }
        .score-item { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 10px 0; 
            border-bottom: 1px solid #eee;
        }
        .score-item:last-child { border-bottom: none; }
        .score-value { 
            font-weight: bold; 
            font-size: 1.2em; 
            color: #667eea;
        }
        .suggestions { 
            background: #e8f5e9; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 15px 0;
        }
        .suggestions h3 { color: #2e7d32; margin-bottom: 15px; }
        .suggestions ul { padding-left: 20px; }
        .suggestions li { margin: 8px 0; color: #333; }
        .error { 
            background: #ffebee; 
            color: #c62828; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0;
        }
        .file-info { 
            background: #e3f2fd; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0;
        }
        .audio-player {
            width: 100%;
            margin: 15px 0;
        }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .dimension-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        .dimension-card:hover {
            transform: translateY(-5px);
        }
        .dimension-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .dimension-score {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .dimension-comment {
            font-size: 0.9em;
            opacity: 0.9;
            line-height: 1.4;
        }
        .overall-score {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        }
        .overall-score h2 {
            font-size: 3em;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .overall-score p {
            font-size: 1.2em;
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .sentences-section {
            background: #f8f9ff;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
        }
        .sentence-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #4facfe;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .sentence-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.9em;
            color: #666;
        }
        .sentence-text {
            font-size: 1.1em;
            line-height: 1.5;
            color: #333;
        }
        .feedback-section {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
        }
        .feedback-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            text-align: center;
        }
        .feedback-content {
            background: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .json-preview {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            max-height: 400px;
            overflow-y: auto;
            position: relative;
        }
        .json-toggle {
            background: #4a5568;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .json-toggle:hover {
            background: #2d3748;
        }
        .tab-container {
            margin: 20px 0;
        }
        .tab-buttons {
            display: flex;
            background: #f1f5f9;
            border-radius: 10px 10px 0 0;
            overflow: hidden;
        }
        .tab-button {
            flex: 1;
            padding: 15px;
            background: #f1f5f9;
            border: none;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            color: #64748b;
            transition: all 0.3s ease;
        }
        .tab-button.active {
            background: white;
            color: #4facfe;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        .tab-content {
            background: white;
            padding: 25px;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .play-btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.8em;
            margin-left: 10px;
            transition: transform 0.2s;
        }
        .play-btn:hover {
            transform: scale(1.05);
        }
        .play-btn:active {
            transform: scale(0.95);
        }
        .tts-info {
            background: #e8f5e9;
            padding: 15px;  
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #4caf50;
        }
        .tts-info h4 {
            color: #2e7d32;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 AI英语口语评估专家</h1>
            <p>基于Qwen2.5 Omni的专业口语分析系统 | AI智能生成评估建议</p>
        </div>
        
        <div class="content">
            <form id="evaluationForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <div id="uploadContent">
                        <h3>📁 上传音频文件</h3>
                        <p>支持格式: WAV, MP3, FLAC, M4A, OGG, WebM</p>
                        <p>最大文件大小: 50MB</p>
                        <br>
                        <button type="button" class="upload-btn" onclick="document.getElementById('audioFile').click()">
                            选择文件
                        </button>
                        <input type="file" id="audioFile" name="audio_file" class="file-input" accept=".wav,.mp3,.flac,.m4a,.ogg,.webm" onchange="handleFileSelect(this)">
                        <p style="margin-top: 15px; color: #666;">或拖拽文件到此区域</p>
                    </div>
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;">
                    <h4>📄 已选择文件:</h4>
                    <p id="fileName"></p>
                    <p id="fileSize"></p>
                    <audio id="audioPreview" class="audio-player" controls style="display: none;"></audio>
                </div>
                
                <!-- 移除评估类型选择，固定使用专业评估 -->
                <input type="hidden" id="promptType" name="prompt_type" value="professional_expert_evaluation">
                
                <div style="display: flex; gap: 10px;">
                    <button type="submit" class="evaluate-btn" id="evaluateBtn" style="flex: 1;">
                        🚀 开始评估
                    </button>
                    <button type="button" class="evaluate-btn" id="cleanupBtn" onclick="cleanupFiles()"
                            style="flex: 0 0 150px; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">🗑️ 清理文件</button>
                </div>
            </form>
            
            <div id="resultArea" class="result-area">
                <div id="loadingDiv" class="loading">
                    <div class="spinner"></div>
                    <h3>🤖 AI正在分析您的口语...</h3>
                    <p>这可能需要1-2分钟，请耐心等待</p>
                </div>
                
                <div id="resultContent" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // 文件拖拽处理
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('audioFile');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            uploadArea.classList.add('dragover');
        }
        
        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(fileInput);
            }
        }
        
        // 文件选择处理
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                const fileName = document.getElementById('fileName');
                const fileSize = document.getElementById('fileSize');
                const fileInfo = document.getElementById('fileInfo');
                const audioPreview = document.getElementById('audioPreview');
                
                fileName.textContent = file.name;
                fileSize.textContent = `大小: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
                
                // 显示音频预览
                const url = URL.createObjectURL(file);
                audioPreview.src = url;
                audioPreview.style.display = 'block';
                
                fileInfo.style.display = 'block';
            }
        }
        
        // 表单提交处理
        document.getElementById('evaluationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files[0]) {
                alert('请先选择音频文件！');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio_file', fileInput.files[0]);
            formData.append('prompt_type', document.getElementById('promptType').value);
            
            // 显示加载状态
            document.getElementById('resultArea').style.display = 'block';
            document.getElementById('loadingDiv').style.display = 'block';
            document.getElementById('resultContent').style.display = 'none';
            document.getElementById('evaluateBtn').disabled = true;
            
            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // 隐藏加载状态
                document.getElementById('loadingDiv').style.display = 'none';
                
                if (response.ok) {
                    displayResult(result);
                } else {
                    displayError(result.error || '评估失败');
                }
                
            } catch (error) {
                document.getElementById('loadingDiv').style.display = 'none';
                displayError('网络错误: ' + error.message);
            } finally {
                document.getElementById('evaluateBtn').disabled = false;
            }
        });
        
        // 显示评估结果
        function displayResult(result) {
            const resultContent = document.getElementById('resultContent');

            let html = `
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button active" onclick="switchTab('scores')">📊 评分结果</button>
                        <button class="tab-button" onclick="switchTab('sentences')">📝 分句分析</button>
                        <button class="tab-button" onclick="switchTab('feedback')">💡 改进建议</button>
                        <button class="tab-button" onclick="switchTab('json')">🔍 JSON预览</button>
                    </div>

                    <div id="scores-tab" class="tab-content active">
                        ${generateScoresTab(result)}
                    </div>

                    <div id="sentences-tab" class="tab-content">
                        ${generateSentencesTab(result)}
                    </div>

                    <div id="feedback-tab" class="tab-content">
                        ${generateFeedbackTab(result)}
                    </div>

                    <div id="json-tab" class="tab-content">
                        ${generateJsonTab(result)}
                    </div>
                </div>
            `;

            resultContent.innerHTML = html;
            resultContent.style.display = 'block';
        }

        // 生成评分标签页
        function generateScoresTab(result) {
            let html = '';

            // 总体评分
            if (result.overall_score) {
                html += `
                    <div class="overall-score">
                        <h2>${result.overall_score}/10</h2>
                        <p>总体评分</p>
                    </div>
                `;
            }

            // 各维度评分
            html += '<div class="score-grid">';

            const dimensions = [
                { key: 'fluency', name: '流利度', icon: '🗣️' },
                { key: 'pronunciation_accuracy', name: '发音准确性', icon: '🎯' },
                { key: 'grammatical_accuracy', name: '语法正确性', icon: '📚' },
                { key: 'lexical_resource', name: '词汇使用', icon: '📖' },
                { key: 'coherence_cohesion', name: '连贯性与逻辑性', icon: '🔗' },
                { key: 'task_fulfillment', name: '内容完整性', icon: '✅' }
            ];

            dimensions.forEach(dim => {
                if (result[dim.key]) {
                    const score = result[dim.key].score || 0;
                    const comment = result[dim.key].comment || '';
                    html += `
                        <div class="dimension-card">
                            <div class="dimension-title">${dim.icon} ${dim.name}</div>
                            <div class="dimension-score">${score}</div>
                            <div class="dimension-comment">${comment}</div>
                        </div>
                    `;
                }
            });

            html += '</div>';

            // 综合评价
            if (result.overall_comment) {
                html += `
                    <div class="feedback-section">
                        <div class="feedback-title">📋 综合评价</div>
                        <div class="feedback-content">${result.overall_comment}</div>
                    </div>
                `;
            }

            return html;
        }

        // 生成分句分析标签页
        function generateSentencesTab(result) {
            let html = `
                <div class="sentences-section">
                    <h3>📝 句子分析</h3>
            `;

            // 检查是否有TTS文件
            const hasTTS = result.tts_files && result.tts_files.sentences;

            // 使用sentence_analysis而不是sentences
            if (result.sentence_analysis && result.sentence_analysis.length > 0) {
                result.sentence_analysis.forEach((sentence, index) => {
                    // 查找对应的TTS音频文件
                    let audioFile = null;
                    if (hasTTS && result.tts_files.sentences && result.tts_files.sentences[index]) {
                        audioFile = result.tts_files.sentences[index].audio_file;
                    }

                    html += `
                        <div class="sentence-item">
                            <div class="sentence-meta">
                                <span>句子 ${sentence.index}</span>
                                <span>词数: ${sentence.word_count || 0} | 字符数: ${sentence.character_count || 0} | 复杂度: ${sentence.complexity || '未知'}</span>
                                ${audioFile ? `<button onclick="playAudio('${audioFile}')" class="play-btn">🔊 播放</button>` : 
                        result.tts_error ? '<span style="color: #e74c3c; margin-left: 10px;">❌ TTS生成失败</span>' : 
                        '<span style="color: #999; margin-left: 10px;">无音频</span>'}
                            </div>
                            <div class="sentence-text">"${sentence.text}"</div>
                            ${audioFile ? `<audio id="audio-${index}" src="/tts_audio/${audioFile.split('/').pop()}" preload="none"></audio>` : ''}
                        </div>
                    `;
                });
            } else {
                html += '<p>暂无分句分析数据</p>';
            }

            html += '</div>';

            // 转录文本
            if (result.transcription) {
                // 查找完整文本的TTS音频
                let fullAudioFile = null;
                if (result.tts_files && result.tts_files.full_transcription) {
                    fullAudioFile = result.tts_files.full_transcription.audio_file;
                }

                html += `
                    <div class="feedback-section">
                        <div class="feedback-title">
                            📄 完整转录文本
                            ${fullAudioFile ? `<button onclick="playAudio('${fullAudioFile}')" class="play-btn" style="margin-left: 10px;">🔊 播放完整文本</button>` : '<span style="color: #999; margin-left: 10px;">无音频</span>'}
                        </div>
                        <div class="feedback-content">${result.transcription}</div>
                        ${fullAudioFile ? `<audio id="audio-full" src="/tts_audio/${fullAudioFile.split('/').pop()}" preload="none"></audio>` : ''}
                    </div>
                `;
            }

            return html;
        }

        // 生成反馈建议标签页
        function generateFeedbackTab(result) {
            let html = `
                <div class="feedback-section">
                    <div class="feedback-title">💡 个性化建议</div>
                    <div class="feedback-content">
            `;

            if (result.personalized_feedback) {
                html += result.personalized_feedback;
            } else {
                html += '暂无个性化建议';
            }

            html += `
                    </div>
                </div>
            `;

            return html;
        }

        // 生成JSON预览标签页
        function generateJsonTab(result) {
            return `
                <button class="json-toggle" onclick="copyJson()">📋 复制JSON</button>
                <div class="json-preview" id="jsonContent">
                    <pre>${JSON.stringify(result, null, 2)}</pre>
                </div>
            `;
        }

        // 切换标签页
        function switchTab(tabName) {
            // 隐藏所有标签页
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // 显示选中的标签页
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }

        // 复制JSON
        function copyJson() {
            const jsonContent = document.getElementById('jsonContent').textContent;
            navigator.clipboard.writeText(jsonContent).then(() => {
                alert('JSON已复制到剪贴板！');
            });
        }

        // 清理文件
        async function cleanupFiles() {
            if (!confirm('确定要清理所有上传的文件和评估结果吗？此操作不可撤销！')) {
                return;
            }

            const cleanupBtn = document.getElementById('cleanupBtn');
            const originalText = cleanupBtn.innerHTML;
            cleanupBtn.innerHTML = '🔄 清理中...';
            cleanupBtn.disabled = true;

            try {
                const response = await fetch('/api/cleanup', {
                    method: 'POST'
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    alert(`✅ ${result.message}`);

                    // 清理界面
                    document.getElementById('fileInfo').style.display = 'none';
                    document.getElementById('resultArea').style.display = 'none';
                    document.getElementById('audioFile').value = '';

                    const audioPreview = document.getElementById('audioPreview');
                    if (audioPreview.src) {
                        URL.revokeObjectURL(audioPreview.src);
                        audioPreview.src = '';
                        audioPreview.style.display = 'none';
                    }
                } else {
                    alert(`❌ 清理失败: ${result.error || '未知错误'}`);
                }

            } catch (error) {
                alert(`❌ 清理失败: ${error.message}`);
            } finally {
                cleanupBtn.innerHTML = originalText;
                cleanupBtn.disabled = false;
            }
        }
        
        // 显示错误信息
        function displayError(message) {
            const resultContent = document.getElementById('resultContent');
            resultContent.innerHTML = `<div class="error">❌ ${message}</div>`;
            resultContent.style.display = 'block';
        }
        
        // 播放音频函数
        function playAudio(audioFile) {
            try {
                // 检查audioFile是否有效
                if (!audioFile) {
                    alert('音频文件不可用');
                    return;
                }
                
                // 停止当前播放的音频
                const allAudioElements = document.querySelectorAll('audio');
                allAudioElements.forEach(audio => {
                    audio.pause();
                    audio.currentTime = 0;
                });
                
                // 从audioFile中提取文件名
                const filename = audioFile.split('/').pop();
                
                // 构造正确的URL路径
                const audioUrl = '/tts_audio/' + filename;
                
                // 创建新的音频对象并播放
                const audio = new Audio(audioUrl);
                audio.play()
                    .then(() => {
                        console.log('音频播放成功: ' + audioUrl);
                    })
                    .catch(error => {
                        console.error('音频播放失败:', error);
                        alert('音频播放失败: ' + error.message + '\n文件路径: ' + audioUrl);
                    });
            } catch (error) {
                console.error('播放音频失败:', error);
                alert('播放音频失败: ' + error.message);
            }
        }
        
        // 获取中文名称
        function getChineseName(key) {
            const names = {
                'pronunciation': '发音',
                'fluency': '流利度',
                'grammar': '语法',
                'vocabulary': '词汇',
                'content': '内容',
                'immediate_actions': '立即行动',
                'practice_methods': '练习方法',
                'learning_resources': '学习资源'
            };
            return names[key] || key;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """音频评估API"""
    try:
        # 检查文件
        if 'audio_file' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'不支持的文件格式，支持格式: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename) # type: ignore
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取参数
        prompt_type = request.form.get('prompt_type', 'professional_expert_evaluation')
        
        logger.info(f"开始评估音频文件: {filename}")
        
        # 调用增强版推理脚本
        cmd = [
            'python', '/home/huangshiang/Eng-speak-correction/qwen_omni_local/enhanced_inference.py',
            '--input', filepath,
            '--type', 'audio',
            '--prompt', prompt_type,
            '--config', '/home/huangshiang/Eng-speak-correction/qwen_omni_local/prompts.yml'
        ]
        
        # 执行评估
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"评估失败: {result.stderr}")
            return jsonify({'error': f'评估失败: {result.stderr}'}), 500
        
        # 查找生成的结果文件
        result_files = [f for f in os.listdir('.') if f.startswith('evaluation_result_') and f.endswith('.json')]
        if not result_files:
            return jsonify({'error': '没有找到评估结果文件'}), 500
        
        # 读取最新的结果文件
        latest_result_file = sorted(result_files)[-1]
        with open(latest_result_file, 'r', encoding='utf-8') as f:
            evaluation_result = json.load(f)

        # 添加TTS功能
        logger.info("🎵 开始生成TTS音频文件...")
        try:
            evaluation_result = tts_service.convert_evaluation_result(evaluation_result)
            logger.info("✅ TTS音频生成完成")
        except Exception as e:
            logger.warning(f"⚠️ TTS音频生成失败: {e}")
            evaluation_result["tts_error"] = str(e)
        
        # 移动结果文件到results目录
        result_filename = f"{timestamp}_result.json"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        os.rename(latest_result_file, result_path)
        
        # 清理上传的文件（可选）
        # os.remove(filepath)
        
        logger.info(f"评估完成: {filename}")
        return jsonify(evaluation_result)
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': '评估超时，请尝试较短的音频文件'}), 500
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """系统状态"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': UPLOAD_FOLDER,
        'results_folder': RESULTS_FOLDER,
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """访问结果文件"""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/tts_audio/<filename>')
def tts_audio_file(filename):
    """访问TTS音频文件"""
    return send_from_directory('tts_audio', filename)

@app.route('/api/cleanup', methods=['POST'])
def api_cleanup():
    """清理上传文件和结果文件"""
    try:
        # 清理上传文件
        upload_files = []
        if os.path.exists(UPLOAD_FOLDER):
            upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
            for file in upload_files:
                os.remove(os.path.join(UPLOAD_FOLDER, file))

        # 清理结果文件
        result_files = []
        if os.path.exists(RESULTS_FOLDER):
            result_files = [f for f in os.listdir(RESULTS_FOLDER) if os.path.isfile(os.path.join(RESULTS_FOLDER, f))]
            for file in result_files:
                os.remove(os.path.join(RESULTS_FOLDER, file))

        # 清理TTS音频文件
        tts_files = []
        tts_audio_dir = 'tts_audio'
        if os.path.exists(tts_audio_dir):
            tts_files = [f for f in os.listdir(tts_audio_dir) if os.path.isfile(os.path.join(tts_audio_dir, f)) and f.endswith('.wav')]
            for file in tts_files:
                os.remove(os.path.join(tts_audio_dir, file))

        # 清理当前目录下的评估结果文件
        current_dir_files = [f for f in os.listdir('.') if f.startswith('evaluation_result_') and f.endswith('.json')]
        for file in current_dir_files:
            os.remove(file)

        logger.info(f"清理完成: {len(upload_files)} 个上传文件, {len(result_files)} 个结果文件, {len(tts_files)} 个TTS音频文件, {len(current_dir_files)} 个评估文件")

        return jsonify({
            'success': True,
            'message': f'清理完成！删除了 {len(upload_files + result_files + tts_files + current_dir_files)} 个文件',
            'details': {
                'upload_files': len(upload_files),
                'result_files': len(result_files),
                'tts_files': len(tts_files),
                'evaluation_files': len(current_dir_files)
            }
        })

    except Exception as e:
        logger.error(f"清理失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 启动 Qwen2.5 Omni 口语评估 Web 应用")
    print("🌐 访问地址: http://localhost:5000")
    print("📁 上传目录:", UPLOAD_FOLDER)
    print("📊 结果目录:", RESULTS_FOLDER)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
