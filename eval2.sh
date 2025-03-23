torchrun --nproc-per-node=8 run.py \
    --data MMStar AI2D_TEST  HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_reasoning \
    --quant_stage reasoning

torchrun --nproc-per-node=8 run.py \
    --data MMStar AI2D_TEST  HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_reasoning \
    --quant_stage caption

torchrun --nproc-per-node=8 run.py \
    --data MMStar AI2D_TEST  HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_reasoning \
    --quant_stage summary

torchrun --nproc-per-node=8 run.py \
    --data MMStar AI2D_TEST  HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_4bit \
    --qbits 4 

torchrun --nproc-per-node=8 run.py \
    --data MMStar AI2D_TEST  HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_origin