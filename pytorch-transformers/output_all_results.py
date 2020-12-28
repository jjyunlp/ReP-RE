"""
根据所有epoch的输出，将所有结果整合，并给出最大的dev值及对应的test结果
"""
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the all epoch results.")
parser.add_argument("--task_name", default=None, type=str, required=True,
                    help="SemEval or Tacred")
parser.add_argument("--start_epoch", default=None, type=int, required=True,
                    help="a epoch num")                 
parser.add_argument("--end_epoch", default=None, type=int, required=True,
                    help="a epoch num")                                        
args = parser.parse_args()

data_dir = args.data_dir
task_name = args.task_name.lower()
start_epoch = args.start_epoch
end_epoch = args.end_epoch

evaluate_option = "macro_f1"
if task_name == "tacred":
    evaluate_option = "f1_micro"

for epoch in range(start_epoch, end_epoch + 1):
    filename = os.path.join(data_dir, f"{task_name}_all_epoch_{epoch}")
    print(filename)
    assert os.path.exists(filename)
    with open(filename) as reader:
        json_result = json.load(reader)
        dev_f1 = json_result['dev'][task_name][evaluate_option]
        test_f1 = json_result['test'][task_name][evaluate_option]
        print(dev_f1)
        print(test_f1)

