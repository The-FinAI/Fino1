
from datasets import load_dataset
import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy
"""
The code is based on HuatuoGPT-o1, please cite this if you use this code.

@misc{chen2024huatuogpto1medicalcomplexreasoning,
      title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs}, 
      author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
      year={2024},
      eprint={2412.18925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.18925}, 
}

"""
class GPT:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **additional_args,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response_data = response.json()


        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data['choices'][0]['message']['content']

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    # def retry_call(self, content, additional_args={"max_tokens": 8192}):
    def retry_call(self, content, additional_args={"max_completion_tokens": 8192}):
        return self.call(content, additional_args)

verify_prompt = """<Model Response>
{}
</Model Response>
<Reference Answer>
{}
</Reference Answer>
You are provided with a model-generated response (<Model Response>) and a reference answer (<Reference Answer>). Compare the model response with the reference answer and determine if the final diagnosis is correct. Your task is to simply output "True" if the response is correct, and "False" otherwise."""

verify_prompt = """<Model Response>  
{}  
</Model Response>  

<Reference Answer>  
{}
</Reference Answer>  

You are provided with a model-generated response (<Model Response>) and a reference answer (<Reference Answer>). Compare the model response with the reference answer and determine its correctness. Your task is to simply output "True" if the response is correct, and "False" otherwise."""


query_prompt_init = """<question>
{}
</question>

Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method. Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. Note that multiple 'Inner Thinking' steps are required to describe thorough reasoning. Each step should first generate a brief title.
- **'Final Conclusion'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final answer. No title is required here.
- **'Verification'**: At this stage, you verify the conclusion from the "Final Conclusion" step. If the conclusion holds, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below:
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

# generate initial reasoning process

query_prompt_init_with_decompose = """<question>
{}
</question>
<instruction>
{}
</instruction>

Please answer the above <question> using the **Chain of Thought (CoT)** reasoning method. The <instruction> section provides helpful hints such as decomposition steps or code logic that can guide your reasoning process. You may refer to these internally, but **do not explicitly mention them** in your reasoning.
Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. Note that multiple 'Inner Thinking' steps are required to describe thorough reasoning. Each step should first generate a brief title.
- **'Final Conclusion'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final answer. No title is required here.
- **'Verification'**: At this stage, you verify the conclusion from the "Final Conclusion" step. If the conclusion holds, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below:
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""



gen_prompt_rethink_Backtracking = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Exploring_New_Path = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to solving this problem and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Verification = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by conducting a thorough **validation** process to ensure validity and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<question>
{}
</question>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_w_label = """<question>
{}
</question>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. Now, I'll secretly tell you that the labeled answer is "{}", but you must pretend not to know. Your 'Verification' requires careful consideration, and if incorrect, you need to provide new Inner Thinking steps and a new Final Conclusion to ensure the final answer aligns with the correct one.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Question>
{}
</Question>

The <Thought Process> above reflects the model's reasoning based on the <Question>. Your task is to rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process. The new version should:

1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for transitions or validations, such as "hmm," "oh," "also," or "wait."
3. Expand the content, making the reasoning richer, more detailed, and logically clear while still being conversational and intuitive.

Return directly the revised natural thinking in JSON format as follows:
```json
{{
  "NaturalReasoning": "..."
}}
```"""

get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<Question>
{}
</Question>

The <Internal Thinking> represents your internal thoughts about the <Question>. Based on this, generate a rich and high-quality final response to the user. If there is a clear answer, provide it first. Ensure your final response closely follows the <Question>. The response style should resemble GPT-4's style as much as possible. Output only your final response, without any additional content."""

# search strategies
search_strategies = [('Backtracking',gen_prompt_rethink_Backtracking),('Exploring New Paths',gen_prompt_rethink_Exploring_New_Path),('Verification',gen_prompt_rethink_Verification),('Correction',gen_prompt_rethink_Correction)]



def extract_bracket_content(text):
        # Extract content between the first '{' and the last '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

def parse_gpt_response(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))
        assert isinstance(da["CoT"],list), "CoT should be list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None

def parse_gpt_response_reformat(response):
    print(response)
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))

        assert isinstance(da["NaturalReasoning"],str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have \\n"
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None 


def get_stream_of_search(longcot):
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'],x['content']))
        else:
            resstr.append(temp.format(x['action'].replace('Final Conclusion','Conclusion'),x['content']))
    return '\n'.join(resstr).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='TheFinAI/Fino1_long_QA_docfinqa', help="Path to the input JSON data file.")

    parser.add_argument("--model_name", type=str, default='gpt-4o', help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, default='********************', help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default='https://api.openai.com/v1/chat/completions', help="OpenAI API URL.")

    parser.add_argument("--max_search_attempts", type=int, default=1, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth.")
    parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")

    parser.add_argument("--limit_num", type=int, default=5, help="Limit the number of processed items.")
    # parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    
    args = parser.parse_args()

    def filter_data(tmpdata):
        filtered_data = []
        for da in tmpdata:
            if 'Open-ended Verifiable Question' not in da or 'Ground-True Answer' not in da:
                continue
            filtered_data.append(da)

        print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
        return filtered_data

    # with open(args.data_path) as f:
    #     tmpdata = json.load(f)
    tmpdata = load_dataset(args.data_path)
    tmpdata = tmpdata["train"]
    tmpdata = [
        {
            "Open-ended Verifiable Question":record["Open-ended Verifiable Question"], #  record['Context and Question'],#
            "Ground-True Answer": record["Ground-True Answer"],
            "Instruction": record['conver'],
            "source": record['source'],
        }
        for record in tmpdata
    ]

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    if args.limit_num:
        data = data[:args.limit_num]
        
    print(f"read data:{len(data)}")

    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_CoT_search'
    save_dir = f'output_data/{task_name}'

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key)


    def verify_gpt(conclusion,answer,d):
        query = verify_prompt.format(conclusion,answer)
        response = gpt_instance.retry_call(query)
        d['gpt4_query_cot'].append(query)
        d['gpt4_response_cot'].append(response)
        if 'true' in response.lower():
            d['verify'].append(True)
            return True
        else:
            d['verify'].append(False)
            return False
        
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 1
            d['verify'] = []
            d['Long_CoT'] = []
            d['gpt4_query_cot'] = []
            d['gpt4_response_cot'] = []
            d['response_struct'] = []
            d['response_type'] = []
            d['prior_fail_try'] = []

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            # init reason
            #query = query_prompt_init.format(d['Open-ended Verifiable Question'])
            if d['Instruction'] != 'No':
                query = query_prompt_init_with_decompose.format(d['Open-ended Verifiable Question'], d['Instruction'])
            else:
                query = query_prompt_init.format(d['Open-ended Verifiable Question'])
            d['gpt4_query_cot'].append(query)
            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                if ii == 0:
                    d['gpt4_response_cot'].append(response)
                flag, struct = parse_gpt_response(response)
                if flag:
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] =  struct["CoT"]
                    d['response_type'].append('Init_CoT')
                    break
                else:
                    print(f'retrying Init_CoT',flush=True)
            if not flag:
                raise Exception('init error')

            verify_gpt(d['Long_CoT'][-2]['content'],d['Ground-True Answer'],d)

            for rethinking_try_time in range(args.max_search_attempts):
                if rethinking_try_time > 0:
                    # Archive the failed state
                    del d['prior_fail_try']
                    save_d['prior_fail_try'].append(d)
                    # Replace with a new state
                    d = save_d

                # Save the initial state
                save_d = copy.deepcopy(d)

                # Begin search
                for rethink_time in range(args.max_search_depth):
                    if d['verify'][-1]:
                        break
                    reasoning = json.dumps(d['Long_CoT'][:-1],ensure_ascii=False,indent=2)
                    # Search strategy
                    if rethink_time > 0:
                        strategy_name,strategy = random.choice(search_strategies)
                    else:
                        # exclude Backtracking
                        strategy_name,strategy = random.choice(search_strategies[1:])

                    query = strategy.format(d['Open-ended Verifiable Question'],reasoning)
                    d['gpt4_query_cot'].append(query)
                    
                    for ii in range(retry_time):
                        response = gpt_instance.retry_call(query)
                        flag, struct = parse_gpt_response(response)

                        if flag:
                            d['gpt4_response_cot'].append(response)
                            d['response_struct'].append(struct["CoT"])
                            d['Long_CoT'] =  d['Long_CoT'][:-1] + struct["CoT"]
                            d['response_type'].append(f'Re_CoT_{strategy_name}')
                            break
                        else:
                            print(f'retrying strategy {strategy_name}',flush=True)
                    if not flag:
                        raise Exception('rethink error')
                    verify_gpt(d['Long_CoT'][-2]['content'],d['Ground-True Answer'],d)
                
                if d['verify'][-1]:
                    break

            # If it is still incorrect, generate a final Label_CoT round
            if not d['verify'][-1] and args.efficient_search:
                reasoning = json.dumps(d['Long_CoT'][:-1],ensure_ascii=False,indent=2)
                query = gen_prompt_w_label.format(d['Open-ended Verifiable Question'],reasoning,d['Ground-True Answer'])
                d['gpt4_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)       
                    flag, struct = parse_gpt_response(response)
                    if flag:
                        d['gpt4_response_cot'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] =  d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append('Label_CoT')
                        # ignore verify
                        d['verify'].append(True)
                        break
                    else:
                        print(f'retrying Label_CoT',flush=True)
                if not flag:
                    raise Exception('label error') 
            
            if d['verify'][-1]:
                # Generate complex CoT and final response (Complex_CoT, response)
                sos = get_stream_of_search(d['Long_CoT'])
                query = reformat_to_complex_cot_prompt.format(sos,d['Open-ended Verifiable Question'])
                d['gpt4_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response_reformat(response)
                    if flag:
                        d['gpt4_response_cot'].append(response)
                        d["Complex_CoT"] = struct["NaturalReasoning"]
                        # get response
                        query = get_final_response_prompt.format(d['Complex_CoT'],d['Open-ended Verifiable Question'])
                        d['gpt4_query_cot'].append(query)
                        response = gpt_instance.retry_call(query)
                        d['gpt4_response_cot'].append(response)
                        d["Response"] = response
                        d['Question'] = d['Open-ended Verifiable Question']
                        break

            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False,indent=2)
                wrongtime = 0

        except Exception as e:
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 20:
                assert 1 == 0, 'wrong'
        return 1
            
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]


    def merge_saved_files(save_dir):
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json')]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'Complex_CoT' in da and 'Response' in da
                    res.append(da)
            except Exception as e:
                continue
        return res
    
    os.makedirs(save_dir, exist_ok=True)

    # Merge previously processed files
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

     # Merge and save final output
    final_data = merge_saved_files(save_dir)
    output_path = f"{task_name}_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
