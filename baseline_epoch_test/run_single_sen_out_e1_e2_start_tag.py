import os

"""
Add trans from pretrained model:
single sentence pretrained
sentence pair pretrained
"""
def run():        
    num_per_rel_list = ['full']#, '100']#, 'whole']     
    num_train_epochs_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_train_epochs_list = [2, 3]
    cuda_num = "0,1"
    learning_rate_list = [3e-5, 5e-5]
    learning_rate_list = [3e-5]
    model_type = "bert"
    bert_model = "../../BERT_BASE_DIR/bert-base-uncased"
    per_gpu_batch_size_list = [8] # use 2 GPUs, so batch_size = 16, 32 
    task_name_list = ["tacred"]
    seed_list = [0, 1, 2]   
    
    for seed in seed_list:
        print(f"Seed: {seed}")
        for task_name in task_name_list:
            print(f"task name: {task_name}")
            data_dir = f"../data/{task_name}"
            for per_gpu_batch_size in per_gpu_batch_size_list:
                print(f"per gpu batch_size={per_gpu_batch_size}")
                for learning_rate in learning_rate_list:
                    print(f"learning rate: {learning_rate}")
                    for num_per_rel in num_per_rel_list:                
                        # all cache will saved in output_dir, so need the num_per_rel
                        output_dir=f"./output/tacred/baseline"
                        train_file = f"{task_name}_train_tag_wrap_{num_per_rel}.json"
                        dev_file = f"{task_name}_dev_tag_wrap.json"
                        test_file = f"{task_name}_test_tag_wrap.json"            
                        
                        for num_train_epochs in num_train_epochs_list:
                            run_cmd = f"CUDA_VISIBLE_DEVICES={cuda_num} python -u single_sen_out_e1_e2_start_tag.py " \
                                    f"--task_name={task_name} --num_train_epochs={num_train_epochs} " \
                                    f"--train_file={train_file} --dev_file={dev_file} --test_file={test_file} " \
                                    f"--num_per_rel={num_per_rel}  --per_gpu_train_batch_size={per_gpu_batch_size} " \
                                    f"--learning_rate={learning_rate} " \
                                    f"--save_steps=0 --cache_dir={output_dir} " \
                                    f"--overwrite_output_dir " \
                                    f"--data_dir={data_dir} " \
                                    f"--do_train --do_eval --do_test --do_lower_case " \
                                    f"--evaluate_during_training " \
                                    f"--model_type={model_type} " \
                                    f"--model_name_or_path={bert_model} " \
                                    f"--max_seq_length=128 " \
                                    f"--output_dir={output_dir}/{per_gpu_batch_size}_{learning_rate}/{task_name}_epoch{num_train_epochs} "
                            print(run_cmd)
                            os.system(run_cmd)
        

# To use multiple GPUs
if __name__ == "__main__":
    # In command line, run python this.py gpu_id=0/1/2/3
    run()
