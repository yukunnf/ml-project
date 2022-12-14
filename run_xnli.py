from utils.utils import *
from transformers import BertModel, AutoModel

set_seed()
args = get_args()

LR_PRETRAIN = 0.00001
LR_FINETUNE = 0.00009
WARMUP_STEPS = 200
TRAINING_STEPS = 8000
MAX_LENGTH = 128
MODEL_NAME = "xlm-roberta-base"# "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)# local_files_only=True, cache_dir="./data", add_prefix_space=True)#, cache_dir="./data")

LANGUAGE_IDS = ["ar",
                "bg",
                "de",
                "el",
                "es",
                "fr",
                "hi",
                "ru",
                "sw",
                "tr",
                "ur",
                "vi",
                "zh"]

LANGUAGE_IDS_for_aligned_tokens = ["ar",
                "bg",
                "de",
                "el",
                "es",
                "fr",
                "hi",
                "ru",
                "sw",
                "tr",
                "ur",
                "vi",
                "zh"]

aligned_tokens = get_aligned_tokens(LANGUAGE_IDS_for_aligned_tokens)

print(args.ratio, args.mode)

eval_res = {}
for _lg in LANGUAGE_IDS:
    eval_res[_lg] = []


class CrossLingualModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()

        self.xlm = AutoModel.from_pretrained(MODEL_NAME)

        self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)

        self.num_labels = num_labels
        self.ner_classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):

        input_ids = input_ids.to(self.xlm.device)
        attention_mask = attention_mask.to(self.xlm.device)

        outputs = self.xlm(input_ids, attention_mask)["last_hidden_state"][:,0,:]

        sequence_output = self.dropout(outputs)
        logits = self.ner_classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(self.xlm.device)
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(
                -1))

        return loss, logits


def _process_task_data(lg):
    raw_datasets = load_dataset("xnli", lg, split="test",
                                cache_dir="./data/huggingface")

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )

    eval_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    return eval_dataset


def process_task_data():
    eval_dataloaders = {}
    for lg in LANGUAGE_IDS:
        eval_dataloaders[lg] = DataLoader(_process_task_data(lg), collate_fn=default_data_collator,
                                          batch_size=64)

    return eval_dataloaders


def get_train_dataloader():
    raw_datasets = load_dataset("xnli", "en", split="train", keep_in_memory=True,
                                cache_dir="./data/huggingface")

    train_hypothesis = raw_datasets["hypothesis"][:]
    train_premise = raw_datasets["premise"][:]

    train_hypothesis_new = []
    train_premise_new = []
    for i in range(len(train_hypothesis)):
        _train_hypothesis_list = train_hypothesis[i].split()
        for token_i in range(len(_train_hypothesis_list)):
            cur_token = _train_hypothesis_list[token_i].lower()
            if cur_token in aligned_tokens:
                if random.random() < args.ratio:
                    _train_hypothesis_list[token_i] = random.choice(aligned_tokens[cur_token])
        train_hypothesis_new.append(" ".join(_train_hypothesis_list))

        _train_premise_list = train_premise[i].split()
        for token_i in range(len(_train_premise_list)):
            cur_token = _train_premise_list[token_i].lower()
            if cur_token in aligned_tokens:
                if random.random() < args.ratio:
                    _train_premise_list[token_i] = random.choice(aligned_tokens[cur_token])
        train_premise_new.append(" ".join(_train_premise_list))

    raw_datasets = raw_datasets.add_column("new_hypothesis", train_hypothesis_new)
    raw_datasets = raw_datasets.add_column("new_premise", train_premise_new)

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["new_premise"],
            examples["new_hypothesis"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )

    processed_raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    train_dataloader = DataLoader(
        processed_raw_datasets, shuffle=True, collate_fn=default_data_collator, batch_size=32
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
    for epoch in range(2):
        model.train()
        all_loss = 0
        update_step = 0
        #train_dataloader = get_train_dataloader()
        for batch in tqdm(train_dataloader):
            #rint(batch)
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


def evaluate(model, dataloaders):
    for lg, dataloader in dataloaders.items():
        _evaluate(model, dataloader, lg)


def _evaluate(model, dataloader, lg):
    nums = 0
    acc = 0

    for batch in tqdm(dataloader):
        label_ids = batch["labels"].view(-1).tolist()

        _, logits = model(batch["input_ids"], batch["attention_mask"])
        logits = logits.cpu()
        index = torch.argmax(logits, -1).view(-1).tolist()

        nums += len(label_ids)
        acc += sum([1 if p==t else 0 for p, t in zip(index, label_ids)])

    print(lg, acc/nums)
    eval_res[lg].append(acc/nums)


main()

