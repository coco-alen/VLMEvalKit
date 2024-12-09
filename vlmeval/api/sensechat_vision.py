from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import img_root_map
from vlmeval.dataset import DATASET_TYPE

URLS = {
    "test1": "http://101.230.144.192:21207/generate",
}

SYSTEM_PROMPT = "<|im_start|>system\n你是商汤科技开发的日日新多模态大模型，英文名叫Sensechat，是一个有用无害的人工智能助手。<|im_end|>\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<audio></audio>\n"


def handle_url(url):
    if url.startswith("file://"):
        with open(url[7:], "rb") as f:
            return base64.b64encode(f.read()).decode()
    else:
        req = requests.get(url)
        return base64.b64encode(req.content).decode()


class SenseChatVisionWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'SenseChat-5-Vision',
                 retry: int = 5,
                 wait: int = 5,
                 ak: str = None,
                 sk: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.ak = os.environ.get('SENSECHAT_AK', None) if ak is None else ak
        self.sk = os.environ.get('SENSECHAT_SK', None) if sk is None else sk
        assert self.ak is not None and self.sk is not None
        self.max_new_tokens = max_tokens
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)
        img_root = osp.join(ROOT, 'images', img_root_map(dataset))
        os.makedirs(img_root, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def image_to_base64(self, image_path):
        import base64
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def encode_jwt_token(self, ak, sk):
        import jwt
        headers = {'alg': 'HS256', 'typ': 'JWT'}
        payload = {
            'iss': ak,
            'exp': int(time.time())
            + 1800,  # 填写您期望的有效时间，此处示例代表当前时间+30分钟
            'nbf': int(time.time()) - 5,  # 填写您期望的生效时间，此处示例代表当前时间-5秒
        }
        token = jwt.encode(payload, sk, headers=headers)
        return token

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ' and 'MMMU' not in dataset:
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and 'MMMU' in dataset:
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = {
                'multiple-choice': 'Answer with carefully thought step by step. Apply the thinking process recursively at both macro and micro levels. Verify consistency of reasoning and look for potential flaws or gaps during thinking. When realize mistakes, explain why the previous thinking was incorrect, fix it and then continue thinking.\n\n',  # noqa: E501
                'open': 'Answer with carefully thought step by step. Apply the thinking process recursively at both macro and micro levels. Verify consistency of reasoning and look for potential flaws or gaps during thinking. When realize mistakes, explain why the previous thinking was incorrect, fix it and then continue thinking.\n\n'  # noqa: E501
            }  # cot prompt
            subject = '_'.join(line['id'].split('_')[1:-1])
            prompt = prompt[line['question_type']].format(subject, subject) + '\n' + question
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        return message

    def message_to_promptimg(self, message, dataset=None):
        if dataset is None or listinstr(['MMMU', 'BLINK'], dataset):
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [[x['value'] for x in message if x['type'] == 'image'][0]]
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        return prompt, image

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get('dataset', None)

        if dataset is not None and listinstr(['ChartQA_TEST','MathVista_MINI'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        else:
            self.max_num = 6

        if dataset is None:
            pass
        elif listinstr(['AI2D_TEST'], dataset):
            self.max_new_tokens = 10
        elif 'MMMU' in dataset:
            self.max_new_tokens = 4096  # 1024
        elif 'MMBench' in dataset:
            self.max_new_tokens = 100
        elif 'MathVista_MINI' in dataset:
            self.max_new_tokens = 4096

        prompt, image = self.message_to_promptimg(message=inputs, dataset=dataset)

        api_secret_key = self.encode_jwt_token(self.ak, self.sk)  # noqa

        content = [{
            'image_base64': self.image_to_base64(item),
            'image_file_id': '',
            'image_url': '',
            'text': '',
            'text': '',
            'type': 'image_base64'
        } for item in image]

        content.append({
            'image_base64': '',
            'image_file_id': '',
            'image_url': '',
            'text': prompt,
            'type': 'text'
        })

        messages = [{'content': content, 'role': 'user'}]

        url = URLS['test1']
        query = ""
        if messages[0]["role"] == "system":
            if messages[0]["content"] != "":
                query += "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
            messages = messages[1:]
        else:
            query += SYSTEM_PROMPT

        images = []
        for message in messages:
            if message["role"] == "user":
                query += USER_START
                if isinstance(message["content"], list):
                    query_content = ""
                    img_cnt = 0
                    for content in message["content"]:
                        if content["type"] == "image_base64":
                            content["type"] = "image_url"
                        if content["type"] == "image_url":
                            query += IMG_TAG
                            img_cnt += 1
                            images.append(
                                {"type": "base64", "data": content["image_base64"]}
                            )
                        elif content["type"] == "text":
                            query_content = content["text"]
                        else:
                            raise ValueError("type must be text, image_url")
                    if img_cnt >= 2:
                        query += f"用户本轮上传了{img_cnt}张图\n"
                    query += query_content + IM_END
                else:
                    query += message["content"] + IM_END
            elif message["role"] == "assistant":
                query += ASSISTANT_START
                query += message["content"] + IM_END
            else:
                raise ValueError("role must be user or assistant")
        query += ASSISTANT_START
        play_load = {
            "inputs": query,
            "parameters": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "stop_sequences": ["<|im_end|>"],
            },
        }
        multimodal_params = {}

        if images:
            multimodal_params["images"] = images
        if multimodal_params:
            play_load["multimodal_params"] = multimodal_params
        data = json.dumps(play_load)
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url,
            headers=headers,
            data=data,
        )

        time.sleep(1)
        try:
            assert response.status_code == 200
            response = response.json()['generated_text'][0].strip()
            # if self.verbose:
            #     self.logger.info(f'inputs: {query}\nanswer: {response}')
            return 0, response, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error('---------------------------ERROR---------------------------')
                self.logger.error(response.json())
                self.logger.error(err)
                self.logger.error(
                    'api error' + response.json()['error']['message']
                    + str([input['value'] if input['type'] == 'image' else None for input in inputs])
                )
                self.logger.error(f'The input messages are {inputs}.')
            return -1, response.json()['error']['message'], ''


class SenseChatVisionAPI(SenseChatVisionWrapper):

    def generate(self, message, dataset=None):
        return super(SenseChatVisionAPI, self).generate(message, dataset=dataset)
