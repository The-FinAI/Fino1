from huggingface_hub import InferenceClient
from openai import OpenAI
import pandas as pd
from datasets import load_dataset
import argparse
import requests
from functools import partial
from tqdm.auto import tqdm
from retrying import retry
import os
from concurrent.futures import ThreadPoolExecutor


def get_api_config(model_name):
    """Get API configuration for a given model."""
    configs = {
        'o1': {'api_url': 'https://api.openai.com/v1', 'api_key': ''},
        'o3-mini': {'api_url': 'https://api.openai.com/v1', 'api_key': ''},
        'gpt-4o': {'api_url': 'https://api.openai.com/v1', 'api_key': ''},
        'deepseek-ai/DeepSeek-V3': {'api_url': 'https://api-inference.huggingface.co/api/inference-proxy/together', 'api_key': ''},
        'deepseek-ai/DeepSeek-R1': {'api_url': 'https://api-inference.huggingface.co/api/inference-proxy/together', 'api_key': ''},
        'deepseek-chat': {'api_url': 'https://api.deepseek.com', 'api_key': '*****'},
        'deepseek-reasoner': {'api_url': 'https://api.deepseek.com', 'api_key': '*****'},
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not configured. Available models: {list(configs.keys())}")
    return configs[model_name]

class GPT:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        messages = [{ "role": "user", "content": content}]
        if self.model_name in ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"]:
            client = InferenceClient(provider="together", api_key="")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        else:
            client = OpenAI(base_url=self.api_url, api_key=self.api_key)
            completion = client.chat.completions.create(model=self.model_name, messages=messages)
        print(completion.choices[0].message)
        return completion.choices[0].message.content

    @retry(wait_fixed=3000, stop_max_attempt_number=10)
    # def retry_call(self, content, additional_args={"max_tokens": 8192}):
    def retry_call(self, content, additional_args={}):
        return self.call(content, additional_args)

def load_fin_dataset(dataset):
    if dataset in ['finqa',  'dm-simplong',  'XBRL_FinMath_test']:
        dataset = load_dataset(dataset, token='*****')
        df = pd.DataFrame(dataset['test'])
    else:
        raise ValueError('dataset not supported')
    return df
    
def create_prediction_dataset(args):
    df = load_fin_dataset(args.dataset)
    config = get_api_config(args.model)
    model = GPT(
        model_name=args.model,
        api_url=config['api_url'],
        api_key=config['api_key']
    )
    
    print(f"\nGenerating predictions for {args.model}")
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing with {args.model}"):
        print(row['id'])
        prediction = model.retry_call(row['query'])
        predictions.append(prediction)
    df['prediction'] = predictions
    
    return df

def main():
    args = parse_args()
    df = create_prediction_dataset(args)
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    avg_len = df['prediction'].str.len().mean()
    print(f"\nAverage prediction length: {avg_len:.2f} characters")
    df.to_csv(os.path.join(args.output_file_dir, f'{args.model}_{args.dataset}'.replace("/", "@")  + '.csv'), index=False)
    print(f"""\nResults saved to {os.path.join(args.output_file_dir, f'{args.model}_{args.dataset}'.replace("/", "-"))}""")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process dataset with LLM model')
    parser.add_argument('--dataset', type=str, default='dm-simplong', help='Name of the HuggingFace dataset')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-V3', help='Model name to use (e.g., gpt-4 or deepseek)')
    parser.add_argument('--output_file_dir', type=str, default='finllm/output_data', help='Output file name for predictions')
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
