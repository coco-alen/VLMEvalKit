# CUDA_VISIBLE_DEVICES=0 python run.py \
#     --data MMStar MMBench_DEV_[EN/CN]_V11 MMVet MathVista_MINI AI2D_TEST HallusionBench\
#     --model LLaVA-CoT \
#     --work-dir ./result_4bit_all \
#     --qbits 4 \
#     --verbose

CUDA_VISIBLE_DEVICES=2 python run.py \
    --data HallusionBench\
    --model LLaVA-CoT \
    --work-dir ./test \
    --qbits 4 

# torchrun --nproc-per-node=8 run.py \
#     --data MMStar \
#     --model LLaVA-CoT \
#     --work-dir ./result_reasoning \
#     --quant_stage reasoning

# torchrun --nproc-per-node=8 run.py \
#     --data MMStar \
#     --model LLaVA-CoT \
#     --work-dir ./result_caption \
#     --quant_stage caption

# torchrun --nproc-per-node=8 run.py \
#     --data MMStar \
#     --model LLaVA-CoT \
#     --work-dir ./result_summary \
#     --quant_stage summary

# python run.py \
#     --data MMStar MMBench_DEV_[EN/CN]_V11 MMVet MathVista_MINI AI2D_TEST HallusionBench\
#     --model LLaVA-CoT \
#     --work-dir ./result \
#     --verbose

