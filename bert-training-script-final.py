import os, time, random, json, gc, warnings
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import EncoderDecoderModel
from nltk.translate.bleu_score import sentence_bleu

warnings.simplefilter("ignore")

MAX_SEQ_LEN = 128
MAX_EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 5
ADAM_LR = 5e-5
BERT_MODEL = 'bert-base-uncased'

ROOT_PATH = '.'
INFERENCE_MODEL_PATH = None
MODEL_PATH = ROOT_PATH + '/models/{:%y-%m-%d-%H%M}-sl{}-bs{}'.format(
    datetime.now(),
    MAX_SEQ_LEN,
    BATCH_SIZE)
if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)

def get_token_type_ids(tokens):
    assert not len(tokens) > MAX_SEQ_LEN
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
                current_segment_id = 1
    assert current_segment_id ==1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def get_token_ids(tokens, tokenizer, max_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_length-len(token_ids))
    return input_ids

def get_model_inputs(text, ans, ques_ids, tokenizer):
    text_token = tokenizer.tokenize(text)
    ans_token = tokenizer.tokenize(ans)
    ques_token = tokenizer.tokenize(ques_ids)
    if len(text_token) > MAX_SEQ_LEN - 3 - len(ans_token):
        text_token = text_token[:MAX_SEQ_LEN - 3 - len(ans_token)]

    input_tokens = ["[CLS]"] + text_token + ["[SEP]"] + ans_token + ["[SEP]"]
    ques_token = ["[CLS]"] + ques_token + ["[SEP]"]
    input_ids = get_token_ids(input_tokens, tokenizer, MAX_SEQ_LEN)
    attention_mask = [1]*len(input_tokens) + [0] * (MAX_SEQ_LEN - len(input_tokens))
    token_type_ids = get_token_type_ids(input_tokens)

    que_ids = get_token_ids(ques_token, tokenizer, len(ques_token))
    que_ids += [0] * (MAX_SEQ_LEN - len(que_ids))

    assert len(que_ids) == MAX_SEQ_LEN
    return input_ids, attention_mask, token_type_ids, que_ids

class JeopardyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.examples = []
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        with open(data_path) as f:
            json_data = json.load(f)
        for data in json_data:
            self.examples.append({
                'text': data['passages'],
                'ans': data['responses'],
                'ques_ids': data['clues']
            })
        del json_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,idx):
        cur_ex= self.examples[idx]
        return [torch.tensor(i, dtype=torch.long) for i in 
            get_model_inputs(cur_ex['text'], cur_ex['ans'], cur_ex['ques_ids'], self.tokenizer)]

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_bleu_score(orig, pre):
    orig_tok= orig.split()
    pre_tok= pre.split()[:len(orig_tok)]
    ref= [orig_tok]
    score= sentence_bleu(ref, pre_tok)
    return score

def predict(eval_data, device, model, worker=0):
    model.eval()
    
    tokenizer = eval_data.tokenizer
    vocab_size = tokenizer.vocab_size
    eval_data_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=worker)
    
    tqdm_loader = tqdm(eval_data_loader, total= len(eval_data_loader))
    total_acc= AverageMeter()
    total_loss = AverageMeter()
    
    predictions= []
    for batch in tqdm_loader:
        input_ids, attention_mask, token_type_ids, ques_ids = [
            i.to(device, dtype=torch.long) for i in batch
        ]
        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=input_ids,
                token_type_ids=token_type_ids,
                masked_lm_labels=ques_ids
            )[:2]
        
        logits = logits.view(-1, vocab_size)
        logits = logits.detach().cpu().numpy()
        orig_ques = ques_ids.detach().cpu().numpy()
        prediction_raw = np.argmax(logits, axis=1).flatten().squeeze()
        prediction_raw = np.reshape(prediction_raw,(BATCH_SIZE,-1))
        cur_pre= []
        for i in range(orig_ques.shape[0]):
            cur_orignal_ques = tokenizer.decode(list(orig_ques[i]), skip_special_tokens=True)
            this_pred_ques = list(prediction_raw[i])
            try:
                cur_len = this_pred_ques.index(102)
            except ValueError:
                cur_len = len(this_pred_ques) - 1
            this_pred_ques = this_pred_ques[:cur_len+1]
            this_pred_ques = tokenizer.decode(this_pred_ques, skip_special_tokens=True)
            this_acc = get_bleu_score(cur_orignal_ques, this_pred_ques)
            cur_pre.append(this_pred_ques)
            total_acc.update(this_acc)
            total_loss.update(loss.item(), attention_mask.size(0))
        predictions += cur_pre
        tqdm_loader.set_postfix(status='valid', accu=total_acc.avg)
    return total_acc.avg, total_loss.avg, predictions

def train(data_train, data_valid, device):
    seed_val = 69
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    tokenizer = data_train.tokenizer
    vocab_size = tokenizer.vocab_size
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased')
    train_dataloader = DataLoader(data_train,
        batch_size=BATCH_SIZE, 
        num_workers=0)

    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight"
    ]
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not 
                any(nd in n for nd in no_decay)], 
         'weight_decay': 0.001
        },
        {
            'params': [
                p for n, p in param_optimizer if
                any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
    }]    
    optimizer = AdamW(
        optimizer_parameters, 
        lr=ADAM_LR
    )

    num_steps = len(data_train) / BATCH_SIZE * MAX_EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps
    )

    model.to(device)
    last_val_loss = 0
    patience = PATIENCE
    for epoch_i in range(0, MAX_EPOCHS):
        total_loss = AverageMeter()
        total_acc = AverageMeter()
        tqdm_loader = tqdm(train_dataloader, total=len(train_dataloader))
        model.train()
        for batch in tqdm_loader:
            input_ids, attention_mask, token_type_ids, ques_ids = [
                i.to(device, dtype=torch.long) for i in batch
            ]
            model.zero_grad()

            loss, logits= model(
                input_ids= input_ids,
                attention_mask= attention_mask,
                decoder_input_ids= input_ids,
                token_type_ids= token_type_ids,
                masked_lm_labels = ques_ids
            )[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits = logits.view(-1, vocab_size)
            logits = logits.detach().cpu().numpy()
            orig_ques = ques_ids.detach().cpu().numpy()
            prediction_raw = logits.argmax(axis=1).flatten().squeeze()
            prediction_raw = prediction_raw.reshape((BATCH_SIZE,-1))
            for i in range(orig_ques.shape[0]):
                cur_orignal_ques = tokenizer.decode(list(orig_ques[i]), skip_special_tokens=True)
                this_pred_ques = tokenizer.decode(list(prediction_raw[i]), skip_special_tokens=True)
                this_acc = get_bleu_score(cur_orignal_ques, this_pred_ques)
                total_acc.update(this_acc)
            tqdm_loader.set_postfix(accu=total_acc.avg)
            total_loss.update(loss.item(), attention_mask.size(0))
            tqdm_loader.set_postfix(
                status='train',
                epochs=epoch_i,
                loss=total_loss.avg,
                accu=total_acc.avg)
        
        time.sleep(0.5)
        _, val_loss, _ = predict(data_valid, device, model)

        # patience is a virtue
        if epoch_i > 0:
            if (val_loss - last_val_loss) > -0.0001:
                patience -= 1
                if patience == 0:
                  print('No improvement for {} epochs... Breaking.'.format(PATIENCE))
                  break
            else: 
                patience = PATIENCE
                print('Epoch {}: {:.2f} better than {:.2f}. Saving model...'
                    .format(epoch_i, val_loss, last_val_loss))
                torch.save(model, MODEL_PATH + '/best_model_BERT.pt')
        last_val_loss = val_loss

torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} with max_seq_len {}, batch_size {}'
    .format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            MAX_SEQ_LEN,
            BATCH_SIZE))
print('Placing models in {}'.format(MODEL_PATH))
data_train = JeopardyDataset(ROOT_PATH + '/data_train.json')
data_valid = JeopardyDataset(ROOT_PATH + '/data_dev.json')
train(data_train, data_valid, device)
