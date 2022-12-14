from tqdm import trange, tqdm
from transformers import AutoTokenizer
import pickle
import os
import random


class Aligner:
    def __init__(self, src_lang, tgt_lang, data_root=None, tokenizer_model="bert-base-multilingual-cased"):
        self.data_root = "./data" if data_root is None else data_root
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.parallel_data_file_name = "data/parallel-{}-{}.txt".format(self.src_lang, self.tgt_lang)
        self.parallel_index_file_name = "data/aligned-{}-{}-index.txt".format(self.src_lang, self.tgt_lang)
        self.aligned_tokens_file_name = "data/aligned-tokens-{}-{}".format(self.src_lang, self.tgt_lang)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self.max_align_num = 1000000

    def create_parallel_data(self):
        if self.tgt_lang[0] in ["a", "b", "c", "d"] or self.tgt_lang == "el":
            download_filename = "{}-{}".format(self.tgt_lang, self.src_lang)
        else:
            download_filename = "{}-{}".format(self.src_lang, self.tgt_lang)

        download_out_filename = "{}-{}".format(self.src_lang, self.tgt_lang)
        os.system("wget -O ./tmp/{}.zip https://opus.nlpl.eu/download.php?f=CCMatrix/v1/moses/{}.txt.zip".format(
            download_out_filename, download_filename))
        os.system("unzip ./tmp/{}.zip -d ./tmp".format(download_out_filename))

        nums = 0

        with open("./tmp/CCMatrix.{}.{}".format(download_filename, self.src_lang), "r", encoding="utf-8") as fr_src, \
                open("./tmp/CCMatrix.{}.{}".format(download_filename, self.tgt_lang), "r", encoding="utf-8") as fr_tgt, \
                open(self.parallel_data_file_name, "w") as fw:

            for src, tgt in tqdm(zip(fr_src, fr_tgt)):
                if nums >= self.max_align_num:
                    break

                fw.write(" ".join(self.tokenizer.tokenize(src[:-1])).replace(" ##", "") + " ||| " + " ".join(
                    self.tokenizer.tokenize(tgt[:-1])).replace(" ##", "") + " \n")

                nums += 1

    def run_fast_align(self):
        os.system(
            f' ./fast_align/build/fast_align -i {self.parallel_data_file_name} -d -o -v -I 8 > {self.parallel_index_file_name}')

    def create_parallel_match_table(self):
        aligned_dict = {}
        with open(self.parallel_data_file_name, "r") as fr_sent, open(self.parallel_index_file_name, "r") as fr_align:
            sents = fr_sent.readlines()
            aligns = fr_align.readlines()
            for i in trange(len(sents)):
                lg_1, lg_2 = sents[i][:-1].split("|||")
                lg_1_tokens = lg_1.split()
                lg_2_tokens = lg_2.split()

                for index in aligns[i][:-1].split():
                    lg_1_index, lg_2_index = index.split("-")
                    lg_1_index = int(lg_1_index)
                    lg_2_index = int(lg_2_index)

                    lg_1_token = lg_1_tokens[lg_1_index].lower()

                    lg_2_token = lg_2_tokens[lg_2_index]
                    if lg_1_token != lg_2_token:
                        if lg_1_token not in aligned_dict:
                            aligned_dict[lg_1_token] = []
                        aligned_dict[lg_1_token].append(lg_2_token)

        _new_aligned_tokens = {}
        for key, val in aligned_dict.items():
            _new_aligned_tokens[key] = []
            for _ in range(100):
                _new_aligned_tokens[key].append(random.choice(val))

        with open(self.aligned_tokens_file_name, "wb") as fw:
            pickle.dump(_new_aligned_tokens, fw)

    def remove_cache_files(self):
        os.system("rm ./tmp/*")
        os.system("rm {}".format(self.parallel_data_file_name))
        os.system("rm {}".format(self.parallel_index_file_name))


for lg in ["af",
           "ar",
           "bg",
           "bn",
           "de",
           "el",
           "es",
           "et",
           "eu",
           "fa",
           "fi",
           "fr",
           "he",
           "hi",
           "hu",
           "id",
           "it",
           "ja",
           "jv",
           "ko",
           "ml",
           "mr",
           "ms",
           "nl",
           "pt",
           "ru",
           "sw",
           "ta",
           "tl",
           "tr",
           "ur",
           "vi",
           "zh"]:
    aligner = Aligner("en", lg)
    aligner.create_parallel_data()
    aligner.run_fast_align()
    aligner.create_parallel_match_table()
    aligner.remove_cache_files()

