# Eng-speak
英语口语评分  
使用Qwen2.5-Omni运行  
安装地址  
https://github.com/QwenLM/Qwen2.5-Omni  
# Qwen2.5-Omni
<p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/Omni_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen25-Omni-a2505ce0d5514e">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2503.20215">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo ">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni">API</a>
<!-- &nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2.5-vl">PAI-DSW</a> -->
</p>

We release **Qwen2.5-Omni**, the new flagship end-to-end multimodal model in the Qwen series. Designed for comprehensive multimodal perception, it seamlessly processes diverse inputs including text, images, audio, and video, while delivering real-time streaming responses through both text generation and natural speech synthesis. Let's click the video below for more information 😃

<a href="https://youtu.be/yKcANdkRuNI" target="_blank">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/video_cover.png" alt="Open Video"/>
</a>


## News
* 2025.06.12: Qwen2.5-Omni-7B ranked first among open source models in the spoken language understanding and reasoning benchmark [MMSU](https://arxiv.org/abs/2506.04779).
* 2025.06.09: Congratulations to our open source Qwen2.5-Omni-7B for ranking first in the [MMAU](https://sakshi113.github.io/mmau_homepage/#leaderboard) leaderboard, and first in the [MMAR](https://github.com/ddlBoJack/MMAR) of open source models in the audio understanding and reasoning evaluation!
* 2025.05.16: We release 4-bit quantized Qwen2.5-Omni-7B (GPTQ-Int4/AWQ) models that maintain comparable performance to the original version on multimodal evaluations while reducing GPU VRAM consumption by over 50%+. See [GPTQ-Int4 and AWQ Usage](#gptq-int4-and-awq-usage) for details, and models can be obtained from Hugging Face ([GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)) and ModelScope ([GPTQ-Int4](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-AWQ))
* 2025.05.13: [MNN Chat App](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#releases) support Qwen2.5-Omni now, let's experience Qwen2.5-Omni on the edge devices! Please refer to [Deployment with MNN](#deployment-with-mnn) for information about memory consumption and inference speed benchmarks.
* 2025.04.30: Exciting! We We have released Qwen2.5-Omni-3B to enable more platforms to run Qwen2.5-Omni. The model can be downloaded from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Omni-3B). The [performance](#performance) of this model is updated, and please refer to [Minimum GPU memory requirements](#minimum-gpu-memory-requirements) for information about resource consumption. And for best experience, [transformers](#--transformers-usage) and [vllm](#deployment-with-vllm) code have update, you can pull the [official docker](#-docker) again to get them.
* 2025.04.11: We release the new vllm version which support audio ouput now! Please experience it from source or our docker image.
* 2025.04.02: ⭐️⭐️⭐️ Qwen2.5-Omni reaches top-1 on Hugging Face Trending! 
* 2025.03.29: ⭐️⭐️⭐️ Qwen2.5-Omni reaches top-2 on Hugging Face Trending! 
* 2025.03.26: Real-time interaction with Qwen2.5-Omni is available on [Qwen Chat](https://chat.qwen.ai/). Let's start this amazing journey now!
* 2025.03.26: We have released the [Qwen2.5-Omni](https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e). For more details, please check our [blog](https://qwenlm.github.io/blog/qwen2.5-omni/)!


# Eng-speak
英语口语评分  
使用Qwen2.5-Omni运行  
安装地址  
https://github.com/QwenLM/Qwen2.5-Omni  






英语数据集网站  
https://www.kaggle.com/datasets/ziya07/college-oral-english-pronunciation-dataset-coepd?resource=download  

还没做完全，后续继续更新

