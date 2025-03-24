torchrun --nproc-per-node=8 --master_port=29501 run.py \
    --data MMStar AI2D_TEST  HallusionBench \
    --model LLaVA-CoT \
    --work-dir ./result_quantkv_reasoning \
    --quant_kv_stage reasoning

torchrun --nproc-per-node=8 --master_port=29501 run.py \
    --data MMStar AI2D_TEST   HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./result_quantkv_caption \
    --quant_kv_stage caption

torchrun --nproc-per-node=8 --master_port=29501 run.py \
    --data MMStar AI2D_TEST  HallusionBench \
    --model LLaVA-CoT \
    --work-dir ./result_quantkv_summary\
    --quant_kv_stage summary