## Debugging model
# python eval.py --model microsoft/phi-4 --evals=MATH500 --tp=2 --output_file=results_phi4.txt

# python eval.py --model microsoft/phi-4 --evals=AIME,MATH500,GPQADiamond --tp=2 --output_file=results_phi-4.txt
# python eval.py --model microsoft/Phi-3.5-mini-instruct --evals=AIME,MATH500,GPQADiamond --tp=2 --output_file=results_Phi-3.5-mini-instruct.txt


python eval.py --model /storage/abdulw/SkyThought/skythought/train/LLaMA-Factory/outputs/phi3/full/original/checkpoint-20800 --evals=AIME,MATH500,GPQADiamond --tp=2 --output_file=results_Phi-3.5-o1-mini-instruct.txt

python eval.py --model microsoft/Phi-3.5-mini-instruct --evals=GPQADiamond --tp=2 --output_file=results_Phi-3.5-mini-instruct.txt


