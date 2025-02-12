
<!-- Title -->
<h1 align="center">🚀 Fino1</h1>

<p align="center">
  <a href="https://your-paper-link.com">📄 Our Paper</a> |
  <a href="https://your-model-link.com">🤖 Our Model</a>
</p>

---

## 📈 Overview

### 📂 Datasets Used
Here, we used 3 evaluation datasets to assess our Fino1 model

| Dataset | Description |
|---------|-------------|
| **[FinQA](https://your-dataset1-link.com)** | descriptions |
| **[DocMath](https://your-dataset2-link.com)** | descriptions |
| **[XBRL-Math](https://huggingface.co/datasets/TheFinAI/Regulation_XBRL_FinMath_test)** | descriptions |

### 🏆 Models Evaluated
We used 16 state-of-the-art large language models (LLMs) to compare with our Fino1 model

| Model | Description |
|-------|------------|
| **[GPT-4o](https://your-model1-link.com)** | descriptions |
| **[GPT-o1](https://your-model2-link.com)** | descriptions |
| **[GPT-o3-mini](https://your-model3-link.com)** | descriptions |
| **[DeepSeek-V3](https://your-model4-link.com)** | descriptions |
| **[DeepSeek-R1](https://your-model5-link.com)** | descriptions |


### 🧩 Reasoning Path Building
Here is the content of the building reasoning path

### 🏗️ How to Train Fino1
For the training part, we were inspired by [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)
- **Stage 1: Supervised Fine-Tuning (SFT)**
- 
Fine-tune the model on an 8-GPU setup:
```bash
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1.py \
    --model_path [meta-llama/Llama-3.1-8B-Instruct] \
    --data_path [FreedomIntelligence/medical-o1-reasoning-SFT] 
```

- **Stage 2: Reinforcement Learning (RL)**

We provide a simple PPO script using the [trl](https://github.com/huggingface/trl) library. Below is an example for training an 8B model with PPO on an 8-GPU A100 machine. Ensure you first download our [medical verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) as the reward model.

```bash
accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
    --config_file ./configs/deepspeed_zero3.yaml \
	--deepspeed_multinode_launcher standard RL_stage2.py \
    --model_name_or_path [FreedomIntelligence/HuatuoGPT-o1-8B] \
    --reward_model_path [FreedomIntelligence/medical_o1_verifier_3B] \
    --value_model_path [meta-llama/Llama-3.2-3B-Instruct] \
    --dataset_name  [FreedomIntelligence/medical-o1-verifiable-problem]\
    --response_length 1300 \
    --temperature 0.5 \
    --local_rollout_forward_batch_size 8 \
    --num_ppo_epochs 3 \
    --num_mini_batches 1 \
    --total_episodes 20000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --bf16 True \
    --output_dir ./ckpts \
    --save_strategy steps \
    --save_step 20 \
    --save_total_limit 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --kl_coef 0.03 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ppo_medical_o1_8B \
    --num_sample_generations -1 \
    --report_to wandb
```

### 🎯 How to Evaluate Fino1

we  construct our Fino1 cart at [Finben](https://github.com/The-FinAI/FinBen) <br>
we also used [DocMath-Eval](https://github.com/yale-nlp/DocMath-Eval) to evaluate our model's ability


---

## Key Highlights
✅ **contribution1**  
✅ **contribution2**  
✅ **contribution3**  

---

## 🛠️ Updates

- **[2025/02/12]** 🎉 We've trained Fino1 model and evaluated its performance recently 
 

