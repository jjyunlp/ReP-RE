import os


"""
TODO
"""

def run():
    # Through gpu_id to select the lr, means on this gpu, the lr argument is fixed
    """
    print("In run function:")
    print("gpu_id = ", type(gpu_id), gpu_id)
    """
    # add trans from base BERT
    task_name = "tacred"
    print(f"task name: {task_name}")
    data_dir = f"../data/{task_name}"
    max_seq_length = 128
    loss_weight_list = [0.2]        
    model_type = "bert"
    bert_model = "../../bert-base-uncased"
    per_gpu_batch_size_list = [8]
    gradient_accumulation_steps = 1 # by accumulation to large the batch_size
    cuda_num = "0,1"
    learning_rate_list = [5e-5] 
    seed_list = [0, 1, 2]   
    output_dir=f"./output"
    mt_system_list = ["all"]    # "google", "baidu", "xiaoniu",             
    num_train_epochs_list = [2, 3]
    for seed in seed_list:
        print(f"seed: {seed}")                                
        for loss_weight in loss_weight_list:            
            for mt_system in mt_system_list:
                for per_gpu_batch_size in per_gpu_batch_size_list:
                    print(per_gpu_batch_size)
                    for learning_rate in learning_rate_list:                        
                        for num_train_epochs in num_train_epochs_list:
                            run_cmd = f"CUDA_VISIBLE_DEVICES={cuda_num} python -u plus_max_bag_loss_for_human_and_trans_with_weight.py " \
                                    f"--task_name={task_name} --num_train_epochs={num_train_epochs} " \
                                    f"--mt_system={mt_system} --loss_weight={loss_weight} " \
                                    f"--save_steps=0 --per_gpu_train_batch_size={per_gpu_batch_size} " \
                                    f"--overwrite_output_dir --cache_dir={output_dir} " \
                                    f"--data_dir={data_dir} --learning_rate={learning_rate} " \
                                    f"--do_train --do_eval --do_test --do_lower_case " \
                                    f"--evaluate_during_training --gradient_accumulation_steps={gradient_accumulation_steps} " \
                                    f"--model_type={model_type} --seed={seed} " \
                                    f"--model_name_or_path={bert_model} " \
                                    f"--max_seq_length={max_seq_length} " \
                                    f"--output_dir={output_dir}/{per_gpu_batch_size}_{gradient_accumulation_steps}_{learning_rate}_{loss_weight}_{seed}/{task_name}_epoch{num_train_epochs} "
                            print(run_cmd)
                            os.system(run_cmd)

# To use multiple GPUs
if __name__ == "__main__":
    # In command line, run python this.py gpu_id=0/1/2/3
    run()
