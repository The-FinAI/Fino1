
<!-- Title -->
<h1 align="center">🚀 Fino1: On the Transferability of Reasoning LLMs and Reinforcement Learning to Finance</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/TheFinAI/FinCoT">🤗 Training Data</a> |
  <a href="https://arxiv.org/pdf/2502.08127">📄 Arxiv</a> |
  <a href="https://huggingface.co/TheFinAI/Fin-o1-8B">🤖 8B-Model</a>
  <a href="https://huggingface.co/TheFinAI/Fin-o1-14B">🤖 14B-Model</a>
  <a href="https://huggingface.co/spaces/TheFinAI/open-finllm-reasoning-leaderboard">🏆 Leaderboard(FinReason)</a>
</p>

---

## 📈 Overview

### 📂 Datasets Used
Here, we utilized three evaluation datasets to assess the performance of our Fino1 model.

| Dataset | Description |
|---------|-------------|
| **[FinQA](https://huggingface.co/datasets/TheFinAI/FINQA_test_test)** | FinQA is a large-scale dataset for numerical reasoning in finance, featuring expert-annotated QA pairs that require integrating structured and unstructured data from financial reports while handling complex domain-specific terminology. |
| **[DocMath](https://huggingface.co/datasets/yale-nlp/DocMath-Eval)** | DocMath-Eval is a benchmark for evaluating LLMs' numerical reasoning over long specialized documents and tables, with the simpllong subset focusing on reasoning across multi-tiered financial or specialized tables within extended contexts. |
| **[XBRL-Math](https://huggingface.co/datasets/TheFinAI/Regulation_XBRL_FinMath_test)** | XBRL-Math dataset evaluates LLMs' numerical reasoning in XBRL filings, requiring models to interpret structured financial data, US GAAP XBRL tags, equations, and hierarchical numerical relationships for accurate financial analysis. |

### 🏆 Models Evaluated
We compared our Fino1 model against 16 state-of-the-art large language models (LLMs).

| Model | Description |
|-------|------------|
| **[GPT-4o](https://platform.openai.com/docs/models#gpt-4o)** | GPT-4o is OpenAI's versatile, high-intelligence flagship model. It accepts text and image inputs and produces text outputs (including Structured Outputs).  |
| **[GPT-o1](https://platform.openai.com/docs/models#o1)** | The o1 series of models are trained with reinforcement learning to perform complex reasoning. o1 models think before they answer, producing a long internal chain of thought before responding to the user. |
| **[GPT-o3-mini](https://platform.openai.com/docs/models#o3-mini)** | o3-mini is OpenAI's most recent small reasoning model, providing high intelligence at the same cost and latency targets of o1-mini. o3-mini also supports key developer features, like Structured Outputs, function calling, Batch API, and more.  |
| **[GPT-4.5](https://platform.openai.com/docs/models/gpt-4.5-preview)** | GPT-4.5 is the largest and most capable GPT model yet. Its deep world knowledge and better understanding of user intent makes it good at creative tasks and agentic planning. GPT-4.5 excels at tasks that benefit from creative, open-ended thinking and conversation, such as writing, learning, or exploring new ideas.  |
| **[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)** | DeepSeek-V3 is a 671B Mixture-of-Experts (MoE) model with 37B active parameters per token, leveraging Multi-head Latent Attention (MLA) and DeepSeekMoE for efficient training and inference, achieving state-of-the-art performance comparable to closed-source models with stable and cost-effective training. |
| **[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** | Qwen2.5 is the latest series of Qwen LLMs, offering models from 0.5B to 72B parameters with improved knowledge, coding, math, instruction following, structured data handling, long-context support (up to 128K tokens), and multilingual capabilities across 29+ languages. |
| **[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)** | Qwen2.5 is the latest series of Qwen LLMs, offering models from 0.5B to 72B parameters with improved knowledge, coding, math, instruction following, structured data handling, long-context support (up to 128K tokens), and multilingual capabilities across 29+ languages. |
| **[Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)** | Qwen2.5 is the latest series of Qwen LLMs, offering models from 0.5B to 72B parameters with improved knowledge, coding, math, instruction following, structured data handling, long-context support (up to 128K tokens), and multilingual capabilities across 29+ languages. |
| **[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)** | Qwen2.5 is the latest series of Qwen LLMs, offering models from 0.5B to 72B parameters with improved knowledge, coding, math, instruction following, structured data handling, long-context support (up to 128K tokens), and multilingual capabilities across 29+ languages. |
| **[Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)** | QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. |
| **[Qwen2.5-Math-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct)** | Qwen2.5-Math-72B-Instruct is an upgraded open-source mathematical LLM supporting both Chain-of-Thought (CoT) and Tool-integrated Reasoning (TIR) for solving math problems in Chinese and English, offering significant performance improvements over Qwen2-Math. |
| **[DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[Llama3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)** | Meta released the Llama 3 family of 8B and 70B LLMs, optimized for dialogue, outperforming many open-source chat models while prioritizing helpfulness and safety. |
| **[Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)** | The Meta Llama 3.1 collection includes multilingual LLMs (8B, 70B, 405B) optimized for multilingual dialogue, outperforming many open-source and closed chat models on industry benchmarks. |
| **[Llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)** | The Meta Llama 3.3 is a 70B instruction-tuned multilingual LLM optimized for dialogue, outperforming many open-source and closed chat models on industry benchmarks. |
| **[DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)** | Meta released the Llama 3 family of 8B and 70B LLMs, optimized for dialogue, outperforming many open-source chat models while prioritizing helpfulness and safety. |
| **[Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** | The Meta Llama 3.1 collection includes multilingual LLMs (8B, 70B, 405B) optimized for multilingual dialogue, outperforming many open-source and closed chat models on industry benchmarks. |
| **[LIMO](https://huggingface.co/GAIR/LIMO)** | LIMO challenges the conventional wisdom in mathematical reasoning by demonstrating that models can achieve superior performance with significantly less but higher quality training data. |
| **[s1-32B](https://huggingface.co/simplescaling/s1-32B)** | s1 is a reasoning model finetuned from Qwen2.5-32B-Instruct on just 1,000 examples. It matches o1-preview & exhibits test-time scaling via budget forcing. |
| **[FinR1-7B](https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1)** | Fin-R1 is a large language model designed for complex reasoning tasks in the financial domain. It was developed and open-sourced jointly by the Financial Large Language Model Research Group (SUFE-AIFLM-Lab) at the School of Statistics and Data Science of Shanghai University of Finance and Economics, in collaboration with Fintopia. The model is based on Qwen2.5-7B-Instruct and fine-tuned using high-quality, verifiable financial reasoning questions. It has achieved state-of-the-art performance across multiple financial benchmarks. |


### 🧩 Reasoning Path Building
For the reasoning path building and training part, we were inspired by [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)

We release the reasoning path here: [FinCoT](https://huggingface.co/datasets/TheFinAI/FinCoT)

### 🏗️ How to Train Fino1
Refer to [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1), we applied two-stage way to train our Fino1 model
- **Stage 1: Supervised Fine-Tuning (SFT)**
We use [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1) for SFT, please check the code for more training details.
- **Stage 2: Reinforcement Learning (RL)**
We use the code from [open-r1](https://github.com/huggingface/open-r1.git) for the GRPO, please check the code for more training details.


# 🎯 Evaluation of all models

## Inference: Local Models  
Model inference for local models is conducted using **[FinBen](https://github.com/The-FinAI/FinBen)** with the **VLLM framework**.

## Inference: API Models  
For API-based models, evaluation is performed using the **`query_llm.py`** script.

## Evaluation 
For the final evaluation, we used [DocMath-Eval](https://github.com/yale-nlp/DocMath-Eval) to first use GPT to extract final answers from the result and then evaluate the correctness of the answer.

---

## Key Results
### 📊 Performance of Different LLMs on Financial Datasets

| **Models** | **FinQA** | **DocMath-Simplong** | **XBRL-Math** | **DocMath-Complong** | **Average** |
|------------|-----------|----------------------|---------------|----------------------|-------------|
| **GPT-4o** | 72.49 | **60.00** | 72.22 | 39.33 | 61.01 |
| **GPT-o1-preview** | 49.07 | 56.00 | 74.44 | 36.67 | 54.05 |
| **GPT-o3-mini** | 60.87 | 59.00 | 76.67 | 35.00 | 57.89 |
| **DeepSeek-V3** | 73.20 | 53.00 | 76.67 | **42.33** | **61.30** |
| **DeepSeek-R1** | 65.13 | 53.00 | 86.67 | 38.67 | 60.87 |
| **GPT-4.5** | 68.94 | 59.00 | 74.44 | 39.33 | 60.43 |
| **Meta-Llama-4-Scount** | 70.45 | 52.00 | **88.89** | 0.67 | 53.00 |
| **Meta-Llama-3-70B-Instruct** | 58.92 | 41.00 | 56.67 | 13.67 | 42.57 |
| **Llama-3.1-70B-Instruct** | 63.18 | 48.00 | 63.33 | 34.33 | 52.21 |
| **Llama-3.3-70B-Instruct** | 68.15 | 54.00 | 70.00 | 32.00 | 56.04 |
| **Qwen2.5-72B-Instruct** | 73.38 | 59.00 | 67.78 | 14.67 | 53.71 |
| **Qwen2.5-Math-72B-Instruct** | 69.74 | 42.00 | 83.33 | 5.00 | 50.02 |
| **DeepSeek-R1-Distill-Llama-70B** | 66.73 | 53.00 | 86.67 | 30.67 | 59.27 |
| **Qwen2.5-32B-Instruct** | 73.11 | 56.00 | 65.56 | 30.00 | 56.17 |
| **Qwen/QwQ-32B** | 61.22 | 46.00 | 84.44 | 20.00 | 52.92 |
| **DeepSeek-R1-Distill-Qwen-32B** | 65.48 | 55.00 | 84.44 | 24.67 | 57.40 |
| **Limo** | 63.44 | 45.00 | 61.11 | 15.33 | 46.22 |
| **S1-32B** | 66.81 | 53.00 | 84.44 | 24.00 | 57.06 |
| **Qwen2.5-14B-Instruct** | 67.44 | 59.00 | 57.78 | 26.67 | 52.72 |
| **DeepSeek-R1-Distill-Qwen-14B** | 63.27 | 44.00 | 84.44 | 21.00 | 53.18 |
| **DeepSeek-R1-Distill-Llama-8B** | 45.96 | 33.00 | 81.11 | 15.67 | 43.94 |
| **Meta-Llama-3-8B-Instruct** | 41.97 | 29.00 | 48.89 | 6.00 | 31.47 |
| **Llama-3.1-8B-Instruct** | 54.13 | 34.00 | 62.22 | 14.30 | 41.16 |
| **Qwen2.5-7B-Instruct** | 55.37 | 41.00 | 42.22 | 17.67 | 39.07 |
| **FinR1-7B** | 58.74 | 37.00 | 30.00 | 13.67 | 34.85 |
| **Fino1-8B** | 73.03 | 56.00 | 84.44 | 26.33 | 59.95 |
| **Fino1-14B** | **74.18** | 55.00 | 87.78 | 27.33 | 61.07 |




---

## 🛠️ Updates

- **[2025/02/12]** 🎉 We've trained Fino1-8B model and evaluated its performance.
- **[2025/03/30]** 🎉 We've trained Fino1-14B model and evaluated its performance recently
  
---

## 📄 Citation
If you find our work useful, please cite our paper:

**BibTeX:**
```bibtex
@misc{qian2025fino1transferabilityreasoningenhanced,
      title={Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance}, 
      author={Lingfei Qian and Weipeng Zhou and Yan Wang and Xueqing Peng and Jimin Huang and Qianqian Xie},
      year={2025},
      eprint={2502.08127},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08127}, 
}


