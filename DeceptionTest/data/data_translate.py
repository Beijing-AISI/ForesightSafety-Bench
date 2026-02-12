import os
import requests
import random
import json
import copy
from hashlib import md5
import pandas as pd
from tqdm import tqdm


class Translator():
    def __init__(self, from_lang='en', to_lang='zh'):
        self.app_id = 'your_id'
        self.app_key = 'your_key'
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path

    @staticmethod
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    def query(self, query: str) -> str:
        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.app_id + query + str(salt) + self.app_key)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.app_id, 'q': query, 'from': self.from_lang, 'to': self.to_lang, 'salt': salt,
                   'sign': sign}

        # Send request
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()

        try:
            src_content = result["trans_result"][0]["dst"]
        except KeyError:
            print(result["error_msg"])
            print(query)
            print()
            src_content = None

        return src_content


if __name__ == '__main__':
    input_file = 'ToMDataset.jsonl'
    output_file = 'CogToM-en.jsonl'

    translator = Translator(from_lang='zh', to_lang='en')
    translated = []

    done_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            done_count = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'a', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        remaining_lines = lines[done_count:]

        for line in tqdm(remaining_lines, desc="Translating CogToM", total=len(lines), initial=done_count):
            line = line.strip()
            raw_data = json.loads(line)
            new_data = raw_data.copy()

            new_data['scene'] = translator.query(raw_data['scene'])
            new_data['question'] = translator.query(raw_data['question'])
            new_options = {}
            for k, v in new_data['options'].items():
                new_options[k] = translator.query(v)
            new_data['options'] = new_options

            f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            f_out.flush()
