import pandas as pd
from dotenv import load_dotenv
import os
import snowflake.connector
import json
import io
import csv
import math 
import glob
import psycopg2

load_dotenv() 

class QueryEvaluator:
    def __init__(self,args):
        self.args = args
        self.env = {}

        assert os.path.exists(args.condition_path) and args.condition_path.endswith(".jsonl"), \
            f"Invalid condition_path, must be a valid jsonl file: {args.condition_path}"

        self.cond = {}
        with open(args.condition_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                instance_id = obj.get("instance_id")
                if instance_id is not None:
                    self.cond[instance_id] = obj

    def run(self):
        if self.args.env == "snowflake" and self.args.env not in self.env.keys() :
            snowflake_credential = json.load(open(os.environ["SF_CREDENTIAL_PATH"]))
            self.env[self.args.env] = snowflake.connector.connect(**snowflake_credential)

        if self.args.env == "postgre" and self.args.env not in self.env.keys() :
            self.env[self.args.env] = psycopg2.connect(
                    dbname=self.args.db_name,
                    user=self.args.username,
                    password=self.args.password,
                    host=self.args.host,
                    port=self.args.port
                )
            self.env[self.args.env].autocommit = True
            
    def run_query(self, query:str):
        with self.env[self.args.env].cursor() as cursor:
            try:
                cursor.execute(query)
                column_info = cursor.description
                
                # DATA
                rows = cursor.fetchmany(size=self.args.max_query)
                columns = [desc[0] for desc in column_info]

            except Exception as e:
                return pd.DataFrame()
            
            if not rows:
                return pd.DataFrame()
            else:
                return pd.DataFrame(rows, columns=columns)

    def vectors_match(self,v1, v2, tol, ignore_order_=False):
        
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    def load_csv(self,path):
        try:
            gold = pd.read_csv(path)
            if gold.shape[1] == 0:
                gold = pd.DataFrame()
        except pd.errors.EmptyDataError:
            gold = pd.DataFrame()
        except FileNotFoundError:
            gold = pd.DataFrame()

        return gold

    def evaluate_query(self, item , query):
        tolerance = 1e-2
        condition_cols = self.cond[item['instance_id']].get("condition_cols")
        ignore_order = self.cond[item['instance_id']].get("ignore_order")

        single_path = os.path.join(self.args.gold_path, f"{item['instance_id']}.csv")

        multi_table = False
        if os.path.exists(single_path):
            golds = self.load_csv(single_path)
        else:
            pattern = os.path.join(self.args.gold_path, f"{item['instance_id']}_*.csv")
            matching_files = sorted(glob.glob(pattern))
            
            if not matching_files:
                raise FileNotFoundError(f"No files found for base: {pattern}")
            
            golds = [self.load_csv(p) for p in matching_files]
            multi_table = True
        
        pred_df = self.run_query(query)

        if not multi_table:
            if condition_cols != []:
                gold_cols = golds.iloc[:, condition_cols]
            else:
                gold_cols = golds
            pred_cols = pred_df

            t_gold_list = gold_cols.transpose().values.tolist()
            t_pred_list = pred_cols.transpose().values.tolist()
            score = 1
            for _, gold in enumerate(t_gold_list):
                if not any(self.vectors_match(gold, pred, tol = tolerance, ignore_order_=ignore_order) for pred in t_pred_list):
                    score = 0
                else:
                    for j, pred in enumerate(t_pred_list):
                        if self.vectors_match(gold, pred, tol = tolerance, ignore_order_=ignore_order):
                            break
        else:
            if condition_cols == [] or condition_cols == [[]] or condition_cols == [None] or condition_cols == None:
                condition_cols = [[] for _ in range(len(golds))]
            elif len(golds) > 1 and not all(isinstance(sublist, list) for sublist in condition_cols):
                condition_cols = [condition_cols for _ in range(len(golds))]
            ignore_order = [ignore_order for _ in range(len(golds))]

            for i, gold in enumerate(golds):
                if condition_cols != []:
                    gold_cols = gold.iloc[:, condition_cols[i]]
                else:
                    gold_cols = gold

                pred_cols = pred_df.copy()

                t_gold_list = gold_cols.transpose().values.tolist()
                t_pred_list = pred_cols.transpose().values.tolist()
                score = 1
                for _, gold in enumerate(t_gold_list):
                    if not any(self.vectors_match(gold, pred, tol = tolerance, ignore_order_=ignore_order[i]) for pred in t_pred_list):
                        score = 0
                    else:
                        for j, pred in enumerate(t_pred_list):
                            if self.vectors_match(gold, pred, tol = tolerance, ignore_order_=ignore_order[i]):
                                break

                if score: # 1 Gold match is counted as correct
                    break

        return score

    
    def end(self):
        for key, env in list(self.env.items()):
            try:
                if env:
                    env.close()
                    del self.env[key]
            except Exception as e:
                print(f"[WARNING] Closing DB {key} raise error: {e}")
