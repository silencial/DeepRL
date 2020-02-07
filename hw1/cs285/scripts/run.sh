#!/bin/zsh
for i in {0..2100..300}
do
  python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter $i --eval_batch_size 5000 --video_log_freq -1
done

