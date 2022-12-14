from utils.utils import *
from transformers import BertModel

set_seed()
args = get_args()

metric = load_metric("seqeval")

LR_PRETRAIN = 0.00001
LR_FINETUNE = 0.00009
WARMUP_STEPS = 200
TRAINING_STEPS = 8000
MAX_LENGTH = 128
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, cache_dir="./data", use_fast=True, add_prefix_space=True)#, cache_dir="./data")
LANGUAGE_IDS = ["Afrikaans",
                "Arabic",
                "Bulgarian",
                "Dutch",
                "English",
                "Estonian",
                "Finnish",
                "French",
                "German",
                "Greek",
                "Hebrew",
                "Hindi",
                "Hungarian",
                "Indonesian",
                "Italian",
                "Japanese",
                "Kazakh",
                "Korean",
                "Chinese",
                "Marathi",
                "Persian",
                "Portuguese",
                "Russian",
                "Spanish",
                "Tagalog",
                "Tamil",
                "Telugu",
                "Thai",
                "Turkish",
                "Urdu",
                "Vietnamese",
                "Yoruba"]

LANGUAGE_IDS_for_aligned_tokens = ["af",
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
                                   "zh"]
aligned_tokens = get_aligned_tokens(LANGUAGE_IDS_for_aligned_tokens)

print(args.ratio, args.mode)

global_label_list = ["S"+str(n) for n in range(17)]

eval_res = {}
for _lg in LANGUAGE_IDS:
    eval_res[_lg] = []


class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=len(global_label_list)):
        super().__init__()

        self.xlm = BertModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = num_labels
        self.ner_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)

        outputs = self.xlm(input_ids, attention_mask)
        sequence_output = self.dropout(outputs)

        logits = self.ner_classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(
                -1))

        return loss, logits

def _process_task_data(lg):
    raw_datasets = load_dataset("xtreme", "udpos." + lg,
                                cache_dir="./data/huggingface")
    text_column_name = "tokens"
    label_column_name = "pos_tags"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["test"][label_column_name])

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )
    return processed_raw_datasets["test"]


def process_task_data():
    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(_process_task_data(lg), collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


def get_train_dataloader():
    raw_datasets = load_dataset("xtreme", "udpos.English", keep_in_memory=True,
                                cache_dir="./data/huggingface")  # ["train_en", "val_en"])

    train_en_tokens = raw_datasets["train"]["tokens"][:]

    for i in range(len(train_en_tokens)):
        for token_i in range(len(train_en_tokens[i])):
            cur_token = train_en_tokens[i][token_i].lower()
            if cur_token in aligned_tokens:
                if random.random() < args.ratio:
                    train_en_tokens[i][token_i] = random.choice(aligned_tokens[cur_token])
    raw_datasets["train"] = raw_datasets["train"].add_column("new_tokens", train_en_tokens)

    for key in list(raw_datasets.keys()):
        if key != "train":
            del raw_datasets[key]

    label_column_name = "pos_tags"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["new_tokens"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        original_tokenized_inputs = tokenizer(
            examples["tokens"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label_to_id[label[word_idx]])

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        if args.ratio == 0:
            return tokenized_inputs

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = original_tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])

            labels.append(label_ids)
        tokenized_inputs["eng_input_ids"] = original_tokenized_inputs["input_ids"]
        tokenized_inputs["eng_attention_mask"] = original_tokenized_inputs["attention_mask"]
        tokenized_inputs["eng_labels"] = labels

        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        keep_in_memory=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=32
    )

    return train_dataloader


def main():
    eval_dataloaders = process_task_data()

    model = CrossLingualModel()
    model.cuda()
    pretrained_params = []
    finetune_params = []
    for (name, p) in model.named_parameters():
        if "xlm" in name:
            pretrained_params.append(p)
        else:
            finetune_params.append(p)

    optimizer = AdamW(
        [{'params': pretrained_params, 'lr': LR_PRETRAIN}, {'params': finetune_params, 'lr': LR_FINETUNE}])
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TRAINING_STEPS)

    train_dataloader = get_train_dataloader()
    for epoch in range(4):
        model.train()
        all_loss = 0
        update_step = 0
        #train_dataloader = get_train_dataloader()
        for batch in tqdm(train_dataloader):
            loss, _ = model(**batch)
            all_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_step += 1

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, eval_dataloaders)

    with open("res-pos-{}".format(args.mode), "a") as fa:
        fa.write("{}\n".format(args.ratio))

        for lg in eval_res.keys():
            best_res = max(eval_res[lg])
            print(lg, best_res, eval_res[lg].index(best_res))
            fa.write("{}\t".format(best_res))
        fa.write("\n\n")


def compute_metrics(p, lg):
    predictions, labels = p
    label_list = global_label_list

    true_predictions = [
        [label_list[int(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[int(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    res = {"precision": results["overall_precision"],
           "recall": results["overall_recall"],
           "f1": results["overall_f1"],
           "accuracy": results["overall_accuracy"]}
    print(lg)
    print(res)
    eval_res[lg].append(float(res["f1"]))


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        _evaluate(model, dataloader, lg)


def _evaluate(model, dataloader, lg):
    preds = []
    targets = []

    for batch in tqdm(dataloader):
        label_ids = batch["labels"].view(-1, 1).tolist()

        _, logits = model(batch["input_ids"], batch["attention_mask"])
        logits = logits.cpu()
        index = torch.argmax(logits, -1).view(-1, 1).tolist()

        preds += index[:]
        targets += label_ids[:]

    compute_metrics((preds, targets), lg)

main()
