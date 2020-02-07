# Behavior Learning

```bash
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --eval_batch_size 5000 --video_log_freq -1

python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name test_bc_humanoid --n_iter 1 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --eval_batch_size 5000 --video_log_freq -1
```

|                |   Ant   | Humanoid |
| :------------: | :-----: | :------: |
|  Expert Mean   | 4713.65 | 10344.52 |
| Imitation Mean | 2710.78 |  339.88  |
|   Expert Std   |  12.20  |  20.98   |
| Imitation Std  | 1225.05 |  77.87   |

## Hyperparameter

```bash
for i in {0..2100..300}
do
	python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter $i --eval_batch_size 5000 --video_log_freq -1
done
```

![bc_params](results/figs/bc.png)


# DAgger

```bash
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name test_bc_humanoid --n_iter 20 --do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --num_agent_train_steps_per_iter 5000 --eval_batch_size 5000 --video_log_freq -1 --use_gpu
```

![dagger](results/figs/dagger.png)

