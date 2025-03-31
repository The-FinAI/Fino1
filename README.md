
<!-- Title -->
<h1 align="center">🚀 Fino1: On the Transferability of Reasoning-Enhanced LLMs to Finance</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA">🤗 Training Data</a> |
  <a href="https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA_v2">🤗 Training Data version 2</a> |
  <a href="https://arxiv.org/pdf/2502.08127">📄 Arxiv</a> |
  <a href="https://huggingface.co/TheFinAI/Fino1-8B">🤖 8B-Model</a>
  <a href="https://huggingface.co/TheFinAI/Fino1-14B">🤖 14B-Model</a>
  <a href="https://huggingface.co/spaces/TheFinAI/open-finllm-reasoning-leaderboard">🏆 Leaderboard</a>
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
| **[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)** | DeepSeek-V3 is a 671B Mixture-of-Experts (MoE) model with 37B active parameters per token, leveraging Multi-head Latent Attention (MLA) and DeepSeekMoE for efficient training and inference, achieving state-of-the-art performance comparable to closed-source models with stable and cost-effective training. |
| **[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)** | DeepSeek-R1-Zero and DeepSeek-R1 are first-generation reasoning models, with DeepSeek-R1 incorporating cold-start data before RL to improve readability and performance, achieving results comparable to OpenAI-o1 across reasoning tasks, while open-sourced distilled models set new benchmarks for dense models. |
| **[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)** | Qwen2.5 is the latest series of Qwen LLMs, offering models from 0.5B to 72B parameters with improved knowledge, coding, math, instruction following, structured data handling, long-context support (up to 128K tokens), and multilingual capabilities across 29+ languages. |
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


### 🧩 Reasoning Path Building
For the reasoning path building and training part, we were inspired by [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)

We release the reasoning path here: https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA

### 🏗️ How to Train Fino1
Refer to [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1), we applied two-stage way to train our Fino1 model
- **Stage 1: Supervised Fine-Tuning (SFT)**

- **Stage 2: Reinforcement Learning (RL)**

We provide a simple PPO script using the [trl](https://github.com/huggingface/trl) library. Below is an example for training an 8B model with PPO on an 8-GPU A100 machine. Ensure you first download [medical verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) as the reward model.

Please check [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1) for more training details.

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
|------------|----------|----------------|--------------|--------------|-----------|
| **GPT-4o** | 72.49 | 60.00 | 72.22 | ??? | 68.24 |
| **GPT-o1** | 49.07 | 56.00 | 74.44 | ??? | 59.84 |
| **GPT-o3-mini** | 60.87 | 59.00 | 76.67 | ??? | 65.51 |
| **DeepSeek-V3** | 73.20 | 53.00 | 76.67 | ??? | 67.62 |
| **DeepSeek-R1** | 65.13 | 53.00 | 86.67 | ??? | 68.93 |
| **Qwen2.5-72B-Instruct** | 73.38 | 59.00 | 67.78 | ??? | 66.72 |
| **Qwen2.5-72B-Instruct-Math** | 69.74 | 42.00 | 83.33 | ??? | 65.69 |
| **DeepSeek-R1-Distill-Llama-70B** | 66.73 | 53.00 | 86.67 | ??? | 68.80 |
| **Llama3-70B-Instruct** | 58.92 | 41.00 | 56.67 | ??? | 52.20 |
| **Llama3.1-70B-Instruct** | 63.18 | 48.00 | 63.33 | ??? | 58.17 |
| **Llama3.3-70B-Instruct** | 68.15 | 54.00 | 70.00 | ??? | 64.05 |
| **DeepSeek-R1-Distill-Qwen-32B** | 65.48 | 55.00 | 84.44 | ??? | 68.97 |
| **DeepSeek-R1-Distill-Qwen-14B** | 63.27 | 44.00 | 84.44 | ??? | 63.90 |
| **DeepSeek-R1-Distill-Llama-8B** | 45.96 | 33.00 | 81.11 | ??? | 53.36 |
| **Llama3-8B-Instruct** | 41.97 | 29.00 | 48.89 | ??? | 39.95 |
| **Llama3.1-8B-Instruct** | 54.13 | 34.00 | 62.22 | ??? | 50.12 |
| **Fino1-8B** | 60.87 | 40.00 | 82.22 | ??? | 61.03 |



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


