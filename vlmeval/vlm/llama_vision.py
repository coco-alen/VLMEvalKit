import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from transformers import BitsAndBytesConfig

class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        # The total number of GPUs might be odd
        num_layers_per_gpu[-1] = total_cost - sum(num_layers_per_gpu[:-1])
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        rank, world_size = get_rank_and_world_size()
        self.quant_stage = kwargs.get('quant_stage', "")
        self.quant_kv_stage = kwargs.get('quant_kv_stage', "")
        self.linear_quantizer = LinearQuantize()
        del kwargs['quant_stage']
        del kwargs['quant_kv_stage']

        if self.quant_stage == "":
            if '11b' in model_path.lower() and auto_split_flag():
                assert world_size == 1, 'We only support world_size == 1 when AUTO_SPLIT is set for Llama-3.2-11B'
                logging.warning('Currently, we only support to split the 11B model across all GPUs.')
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                ).eval()
            elif '90b' in model_path.lower():
                device_map = self.split_model()
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                ).eval()
            else:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cpu',
                    **kwargs
                ).eval().to("cuda")
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cpu',
                ).eval().to("cuda")
            self.model_quant = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    # bnb_4bit_compute_dtype=torch.int8,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
                **kwargs
            ).eval().to("cuda")

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs = {}
        if 'Instruct' in model_path or 'cot' in model_path or 'CoT' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 400
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Think step by step and finally respond to the question '
                    f"with only the correct option number as \"FINAL ANSWER\"."
                    f"<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i+1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
                self.kwargs['max_new_tokens'] = 128
            else:
                self.kwargs['max_new_tokens'] = 512
        if "cot" in self.model_name or "CoT" in self.model_name:
            self.kwargs['max_new_tokens'] = 2048
        # output = self.model.generate(**inputs, **self.kwargs)
        output = self.custom_generate(inputs, self.kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')
        

    def get_llm_outText(self, gen_ids):
        """
        Print the generated text from the model.
        """
        completion_text = self.processor.decode(gen_ids, skip_special_tokens=True)
        return completion_text

    def top_p_process(self, scores, top_p=1.0):
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -1 :] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores_processed = scores.masked_fill(indices_to_remove, -float("Inf"))
            return scores_processed
        else:
            return scores

    def custom_generate(self, inputs, kwargs):
        # print("\n==============================\n")
        model = self.model
        prefill_input_ids = inputs["input_ids"]
        max_new_tokens = kwargs.get("max_new_tokens", 2048)
        temperature = 0.6
        top_p=0.9
        past_key_values = None
        with torch.no_grad():
            # === Prefill step: Run the prompt through the model to get the initial KV cache ===
            # The 'input_ids' here are the tokens for your prompt.
            outputs = model(**inputs, use_cache=True, past_key_values=past_key_values)
            
            # Retrieve the cached key/value pairs
            past_key_values = outputs.past_key_values
            
            # We'll start our generated sequence with the prefill tokens.
            # (If you are generating multiple sequences per prompt, this example assumes a batch dimension.)
            generated = prefill_input_ids
            # print(self.get_llm_outText(generated[0]))
            # === Loop to generate new tokens one by one ===
            for idx in range(max_new_tokens):

                next_input_ids = generated[:, -1].unsqueeze(-1)

                outputs = model(input_ids=next_input_ids, use_cache=True, past_key_values=past_key_values)
                logits = outputs.logits  # shape: (batch_size, 1, vocab_size)
                past_key_values = outputs.past_key_values  # update the KV cache

                next_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

                # === Sampling step ===
                if temperature > 0.0:
                    next_token_logits = next_token_logits / temperature
                    next_token_logits = self.top_p_process(next_token_logits, top_p=top_p)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)


                # print(self.get_llm_outText(next_token[0]), end="")
                generated = torch.cat((generated, next_token), dim=-1)
                # sign = self.get_llm_outText(generated[0,-4:])
                # sign2 = self.get_llm_outText(generated[0,-5:])
                # if sign == "<SUMMARY>" or sign == "</SUMMARY>" or sign == "<CAPTION>" or sign == "<\CAPTION>" or sign2=="<REASONING>" or sign2=="<\REASONING>":
                #     print("===")
                if self.quant_stage == "summary":
                    if self.get_llm_outText(generated[0,-4:]).strip() == "<SUMMARY>":
                        model = self.model_quant
                    elif self.get_llm_outText(generated[0,-4:]).strip() == "</SUMMARY>":
                        model = self.model
                if self.quant_stage == "reasoning":
                    if self.get_llm_outText(generated[0,-5:]).strip() == "<REASONING>":
                        model = self.model_quant
                    elif self.get_llm_outText(generated[0,-5:]).strip() == "</REASONING>":
                        model = self.model
                if self.quant_stage == "caption":
                    if self.get_llm_outText(generated[0,-4:]).strip() == "<CAPTION>":
                        model = self.model_quant
                    elif self.get_llm_outText(generated[0,-4:]).strip() == "</CAPTION>":
                        model = self.model


                if self.quant_kv_stage == "summary":
                    quantKV_count = 0
                    count_start = False
                    if self.get_llm_outText(generated[0,-4:]).strip() == "<SUMMARY>":
                        count_start = True
                    elif self.get_llm_outText(generated[0,-4:]).strip() == "</SUMMARY>":
                        self.quant_lastN_kv_cache(past_key_values, quantKV_count)
                        quantKV_count = 0
                        count_start = False
                    if count_start:
                        quantKV_count += 1
                if self.quant_kv_stage == "reasoning":
                    quantKV_count = 0
                    count_start = False
                    if self.get_llm_outText(generated[0,-5:]).strip() == "<REASONING>":
                        count_start = True
                    elif self.get_llm_outText(generated[0,-5:]).strip() == "</REASONING>":
                        self.quant_lastN_kv_cache(past_key_values, quantKV_count)
                        quantKV_count = 0
                        count_start = False
                    if count_start:
                        quantKV_count += 1
                if self.quant_kv_stage == "caption":
                    quantKV_count = 0
                    count_start = False
                    if self.get_llm_outText(generated[0,-4:]).strip() == "<CAPTION>":
                        count_start = True
                    elif self.get_llm_outText(generated[0,-4:]).strip() == "</CAPTION>":
                        self.quant_lastN_kv_cache(past_key_values, quantKV_count)
                        quantKV_count = 0
                        count_start = False
                    if count_start:
                        quantKV_count += 1

                if (next_token == self.processor.tokenizer.eos_token_id).any():
                    break
        return generated

    def quant_lastN_kv_cache(self, kv_cache, quantKV_count):
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i][:, :, -quantKV_count:, ...] = self.linear_quantizer.quantize(kv_cache.key_cache[i][:, :, -quantKV_count:, ...], 3, 128)
            kv_cache.key_cache[i].contiguous()
            kv_cache.value_cache[i][:, :, -quantKV_count:, ...] = self.linear_quantizer.quantize(kv_cache.value_cache[i][:, :, -quantKV_count:, ...], 3, 128)
            kv_cache.value_cache[i].contiguous()
        return kv_cache


class LinearQuantize:
    def quant_func(self, x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return q.to(torch.uint8)

    def _quantize(self, tensor, nbits, q_group_size):
        batch_size, num_head, seq_len, embedding_dim = tensor.size()
        assert embedding_dim % q_group_size == 0, "embedding_dim must be divisible by q_group_size"
        x = tensor.view(batch_size, num_head, seq_len, embedding_dim // q_group_size, q_group_size).float()

        maxq = torch.tensor(2 ** nbits - 1, device=x.device, dtype=x.dtype)
        tmp = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        xmin = torch.minimum(x.min(-1)[0], tmp)
        xmax = torch.maximum(x.max(-1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        scale = scale.unsqueeze(-1)
        zero = zero.unsqueeze(-1)
        out_int = self.quant_func(x, scale, zero, maxq)
        return {
            "int_tensor": out_int,
            "scale": scale,
            "zero": zero,
            "original_type": tensor.dtype,
        }
    def _dequantize(self, qtensor):
        int_tensor = qtensor["int_tensor"]
        scale = qtensor["scale"]
        zero = qtensor["zero"]
        out_fp = (int_tensor.float() - zero) * scale
        batch_size, num_head, seq_len, n_group, group_size = out_fp.size()
        out_fp = out_fp.view(batch_size, num_head, seq_len, n_group * group_size).to(qtensor["original_type"])
        return out_fp

    def quantize(self, tensor, nbits, q_group_size):
        qtensor = self._quantize(tensor, nbits, q_group_size)
        return self._dequantize(qtensor)