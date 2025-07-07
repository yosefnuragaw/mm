from groq import Groq
from dotenv import load_dotenv
import os
import openai
import google.auth
from google.auth.transport.requests import Request
from google.auth import default
from copy import deepcopy
import time
import pandas as pd
from datetime import datetime


from MM.agent.prompt_builder import get_prompt_builder
from MM.agent.message_processor import MessageProcessor
from MM.agent.evaluator import QueryEvaluator

load_dotenv()

class LLMAgent:
    def __init__(self, args):
        self.args = args
        if self.args.model.startswith("google"):
            credentials,location,project_id = self._initiate_google_cred()

            # Gemini client
            self.client = openai.OpenAI(
                base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
                api_key=credentials.token,
            )
            
        self.prompt_builder = get_prompt_builder(self.args.prompt_strategy)
        self.message_processor = MessageProcessor(args)
        self.evaluator = QueryEvaluator(args)

    def _initiate_google_cred(self):
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials,location,project_id

    def call_llm(self, messages, instance_id=None, round_num=None):
        max_retries = 20
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_new_tokens,
                    n=1,
                )
                
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    raise Exception("Empty response content")
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    return f"ERROR: Failed to get response after {max_retries} retries"
                
                time.sleep(0.2)

            if retry_count % 3 == 0: #if exceed 1 hour
                if self.args.model.startswith("google"):
                    credentials,location,project_id = self._initiate_google_cred()
                    self.client = openai.OpenAI(
                        base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
                        api_key=credentials.token,
                    )
        
        return "ERROR: Unexpected exit from retry loop"
    
    def process_single_item(self, item):
        instance_id = item.get("instance_id")

        try:
            messages = self.prompt_builder.build_initial_prompt(item, self.args)
            conversation_history = deepcopy(messages)
            terminated = False
            round = 0
            for round_num in range(self.args.max_rounds):
                # print(f"Processing {instance_id}, round {round_num + 1}")

                llm_response = self.call_llm(messages, instance_id, round_num + 1)
                
                if llm_response.startswith("ERROR:"):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to get valid LLM response for {instance_id}")
                    error_result = {
                        "instance_id": instance_id,
                        "error": llm_response,
                        "round_failed": round_num + 1,
                        "terminated": False
                    }
                    return error_result
                
                result = self.message_processor.process_round(
                    llm_response, item, messages, conversation_history
                )
                
                round = round_num
                if result.get("terminated"):
                    terminated = True
                    break
                
                if result.get("continue"):
                    continue

            result = {
                "instance_id": instance_id,
                "conversation": conversation_history,
                "final_messages": messages,
                "last_messages" : messages[-1],
                "terminated": terminated
            }
            
            
            status = "TERMINATED" if terminated else "INCOMPLETE"
            # print(f"Completed: {instance_id}/{round}) - {status}")
            
            return result
            
        except Exception as e:
            error_result = {
                "instance_id": instance_id,
                "error": str(e),
                "terminated": False
            }
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error processing {instance_id}: {str(e)}")
            return error_result

    def evaluate(self, items):
        self.evaluator.run()
        
        if not isinstance(items, list):
            items = [items]
        acc = 0
        for it in items:
            result = self.process_single_item(it)
            if result.get("terminated"):

                query = self.message_processor.parse_query(it, result.get("last_messages").get("content"))

                score = self.evaluator.evaluate_query(it, query)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] LLMAgent(name={self.args.model}) | Instance: {it.get('instance_id')} | Scored :{score==1}")
                acc += score
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] LLMAgent(name={self.args.model}) | Instance: {it.get('instance_id')} | Not Terminated!")

        self.evaluator.end()

        return {
            "score": acc/ len(items),
            "correct": acc,
            "false": len(items) - acc,
        }
    
    def __repr__(self):
        return f"LLMAgent(name={self.args.model})"



