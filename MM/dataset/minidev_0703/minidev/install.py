import json
import argparse
import os
import pandas as pd
import psycopg2
import logging
from psycopg2 import OperationalError, errors

def load_json(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f) 

def installer_config()->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Initiate config for installing MMQA dataset"
    )

    #Path settings
    parser.add_argument("--data_path", help="System path", default='MM/dataset/minidev_0703/minidev')
    parser.add_argument("--gold_path", help="Query evaluation path", default='MM/dataset/minidev_0703/minidev/gold')
    parser.add_argument("--condition_path", help="Evaluation condition path", default="MM/dataset/minidev_0703/minidev/MINIDEV/mini_dev_postgresql.json")

    #SQL settings
    parser.add_argument("--db_name", help="DB name", default='bird')
    parser.add_argument("--username", help="SQL username", default='yosef')
    parser.add_argument("--password", help="SQL password", default='yosef')
    parser.add_argument("--host", help="DB host", default="localhost")
    parser.add_argument("--port", help="DB port", default="5432")
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--timeout_ms", type=int, default=60000)
    args = parser.parse_args()

    return args

def parse_task(json):
    return {
        "instance_id": json.get("question_id"),
        "instruction": json.get("question"),
        "db_id": json.get("db_id"),
        "ignore_order": True,
        "condition_cols":[]
    }

def extract_gold(json, args,logger):
    try:
        conn = psycopg2.connect(
            dbname=args.db_name,
            user=args.username,
            password=args.password,
            host=args.host,
            port=args.port
        )
        conn.autocommit = True
        cur = conn.cursor()

        timeout_ms = args.timeout_ms 
        cur.execute(f"SET statement_timeout = {timeout_ms};")

        cur.execute(json.get("SQL"))
        column_info = cur.description
        
        rows = cur.fetchmany(size=args.max_query)
        columns = [desc[0] for desc in column_info]
        if not rows:
            gold = pd.DataFrame()
        else:
            gold = pd.DataFrame(rows, columns=columns)
        
    except errors.QueryCanceled as e:
        logger.warning(f"TIMEOUT on {json.get('question_id')}")
        gold = pd.DataFrame()
        return False
    except Exception as e:
        logger.warning(f"ERROR on {json.get('question_id')}: {e}")
        gold = pd.DataFrame()
        return False

    os.makedirs(args.gold_path, exist_ok=True)
    gold_filepath = os.path.join(args.gold_path,f"{json.get('question_id')}.csv")
    gold.to_csv(gold_filepath, index=False)
    return True

if __name__ == "__main__":
    args = installer_config()

    logging.basicConfig(
    filename=os.path.join(args.data_path, 'installer_log.log'),
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    
    global_task = []
    for task in load_json(args.condition_path):
        if extract_gold(task, args,logger):
            global_task.append(parse_task(task))
        

    file_path = os.path.join(args.data_path, "minibird.jsonl")
    with open(file_path, "w") as f:
        for obj in global_task:
            json_line = json.dumps(obj)
            f.write(json_line + "\n")
