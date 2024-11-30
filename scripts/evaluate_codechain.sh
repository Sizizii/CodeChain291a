split=test_60 # test OR mini_val OR test_subset
model=wizard_coder13B # gpt3.5 or gpt4 
start=0
end=60
num_gen_samples=20
start_round=1
end_round=5

## ROUND 0 
prompt=prompts/codechain_gen.txt 
round=round0
exp_name=${model}_${split}
output_path=outputs/${exp_name}_$round
num_clusters=5

# Test by hidden test cases 
python src/evaluate.py --save_gen_path $output_path --eval_split $split

## REVISION ROUNDS 
for (( round_number = $start_round; round_number <= $end_round; round_number++ )) 
do
    echo "REVISION ROUND $round_number"   
    round=round${round_number}
    output_path=outputs/${exp_name}_$round
    echo "OUTPUT PATH: $output_path"

    # Test by hidden test cases 
    python src/evaluate.py --save_gen_path $output_path --eval_split $split --original_gen_path outputs/${exp_name}_round0
       
done
