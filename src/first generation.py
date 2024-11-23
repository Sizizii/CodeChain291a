import os
import time
import json
import pprint
from tqdm import tqdm
import pickle as pkl
import pandas as pd

from datasets import load_dataset, load_from_disk
import torch
import argparse
from utils.utils import get_util_functions_self_cluster
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Config for generation")

parser.add_argument(
    '--model',
    type=str,
    help="Model name for generation")
parser.add_argument(
    '--output_path',
    type=str,
    help="Output path to save generated code")
parser.add_argument(
    '--prompt_file',
    type=str,
    help="Path to instruction prompt")
parser.add_argument(
    '--modules_file',
    type=str,
    default=None,
    help="Path to extracted modules for self-revision")
parser.add_argument(
    '--num_gen_samples',
    type=int,
    default=20,
    help="Number of generation samples per problem")
parser.add_argument(
    '--split',
    type=str,
    default='test',
    help="name of data split in APPS")
parser.add_argument(
    '--num_clusters',
    type=int,
    default=5,
    help="Number of clusters in prior generation round")
parser.add_argument(
    '--start',
    type=int,
    default=0,
    help="Star index of dataset")
parser.add_argument(
    '--end',
    type=int,
    default=10,
    help="End index of dataset")

args = parser.parse_args()


argsdict = vars(args)
print(pprint.pformat(argsdict))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.split is not None and args.split == 'mini_val':
    apps = load_from_disk(f'../data/{args.split}')
    apps_problem_ls = []
    for level in ['introductory', 'interview', 'competition']:
        apps_problem_ls += list(apps[level])
else:
    apps = load_dataset("codeparrot/apps")[args.split]

with open("../prompts/codechain_gen.txt", 'r', encoding='utf-8') as infile:
    prompt = infile.read()

if not os.path.exists("../outputs/round0"):
    os.makedirs("../outputs/round0", exist_ok=True)

#we don't need modules_files for the first time generation
if args.modules_file is not None:
    if '.csv' in args.modules_file:
        modules = pd.read_csv(args.modules_file)
    else:
        modules = pkl.load(open(args.modules_file, 'rb'))

    modules = get_util_functions_self_cluster(modules, num_clusters=args.num_clusters)
    print("Util functions for {} problems".format(len(modules)))

lens = []
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir='../cache', torch_dtype="auto",
                                          device_map="auto")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir='../cache', torch_dtype="auto",
                                             device_map="auto")
for idx in tqdm(range(args.start, args.end), total=args.end - args.start):
    if args.split is not None and args.split == 'mini_val':
        problem = apps_problem_ls[idx]
    else:
        problem = apps[idx]
    # problem is a dictionary
    problem_id = problem['problem_id']

    if os.path.exists("../outputs" + '/{}.json'.format(problem_id)):
        continue

    question = problem['question']
    curr_prompt = prompt.replace("<<problem>>", question)

    if '<<starter_code>>' in prompt:
        starter_code = problem['starter_code']
        curr_prompt = curr_prompt.replace("<<starter_code>>", starter_code)

    if '<<starter_code_task>>' in prompt:
        starter_code = problem['starter_code']
        if len(starter_code) > 0:
            starter_code_prompt = f"Notes:\nThe final python function should begin with: \n```python\n{starter_code}\n```"
        else:
            starter_code_prompt = ''
        curr_prompt = curr_prompt.replace("<<starter_code_task>>", starter_code_prompt)

    if '<<question_guide>>' in prompt:
        starter_code = problem['starter_code']
        if len(starter_code) > 0:
            question_guide = 'use the provided function signature'
        else:
            question_guide = 'read from and write to standard IO'
        curr_prompt = curr_prompt.replace("<<question_guide>>", question_guide)

    if '<<modules>>' in curr_prompt:
        if problem_id not in modules: continue
        curr_modules = list(modules[problem_id])
        module_seq = ''
        for module in curr_modules:
            module_seq += "\n```module\n" + module.strip() + "\n```\n"
        curr_prompt = curr_prompt.replace('<<modules>>', module_seq)

    success = False
    start = time.time()
    responses = []
    if args.num_gen_samples == 1:
        num_loops = 1
    else:
        num_loops = int(args.num_gen_samples / 5)

    for i in tqdm(range(num_loops), leave=False):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": curr_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
        success = True

    if not success:
        print("Failure to generate! skipp this sample problem id {}".format(problem_id))
        continue

    curr_output = {}
    curr_output['prompt'] = curr_prompt

    for i in range(len(responses)):
        curr_output[f'output{i}'] = responses[i]

    json.dump(curr_output, open("../outputs/round0" + '/{}.json'.format(problem_id), 'w'))