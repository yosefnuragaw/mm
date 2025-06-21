import os
import json
import argparse

from MM.model_selector import ModelSelector
from MM.agent.llm import LLMAgent

def selector_config()->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Initiate config for Model Selector"
    )

    #General
    parser.add_argument("--n_cluster", type=int, default=5)
    parser.add_argument("--sample_rate", type=float, default=0.1)

    #Thompson
    parser.add_argument("--prior_size", type=int, default=16)

    #Succesive Reject
    parser.add_argument("--eval_size", type=int, default=16)
    parser.add_argument("--strategy", default="vanilla", help="['vanilla', 'halving']")
    args = parser.parse_args()

    #UCB
    parser.add_argument("--eval_rate", type=float, default=0.2)
    parser.add_argument("--k_model", type=int, default=2)
    parser.add_argument("--coeff", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=5)

    return args

def model_config(model:str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Initiate config for {model}"
    )
    
    #Path settings
    parser.add_argument("--system_prompt_path", help="System prompt file path", default="MM/agent/prompts/spider_prompt.txt")
    parser.add_argument("--databases_path", help="Databases directory path", default="MM/dataset/spider/resource/databases")
    parser.add_argument("--documents_path", help="Documents directory path", default="MM/dataset/spider/resource/documents")
    parser.add_argument("--gold_path", help="Query evaluation path", default="MM/dataset/spider/gold/exec_result")
    parser.add_argument("--condition_path", help="Evaluation condition path", default="MM/dataset/spider/gold/spider2snow_eval.jsonl")

    #LLM settings
    parser.add_argument("--model", type=str, default = model)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--prompt_strategy", default="spider-agent", 
                       choices=["spider-agent"],
                       help="Prompt building strategy")

    # Execution settings
    parser.add_argument("--api_host", default="localhost", help="API host")
    parser.add_argument("--api_port", default="5000", help="API port")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max conversation rounds")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--rollout_number", type=int, default=1, help="Number of rollouts per example")

    # Evaluation settings
    parser.add_argument("--max_query", type=int , default=1000, help="Number of max row")
    parser.add_argument("--env", default="snowflake", help="Environment for SQL excution ")

    args = parser.parse_args()
    return args

def load_data(test_path):
    assert os.path.exists(test_path) and test_path.endswith(".jsonl"), f"Invalid test_path, must be a valid jsonl file: {test_path}"
    with open(test_path, "r") as f:
        task_configs = [json.loads(line) for line in f]
    
    return task_configs
    
if __name__ == "__main__": #Testing Connection
    plafortm_conf = selector_config()
    configs = [model_config("google/gemini-2.0-flash-001"), model_config("google/gemini-2.5-flash"), model_config("google/gemini-2.5-pro")]
    
    dataset = load_data("MM/dataset/spider/spider2-snow.jsonl")
    models = list()
    for conf in configs:
        models.append(LLMAgent(conf))

    selector = ModelSelector( models=models, args=plafortm_conf)
    res = selector.succesive_reject(dataset)
    print(res)