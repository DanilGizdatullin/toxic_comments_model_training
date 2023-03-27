from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel

from config import Config
from lib.dataset import ToxicCommentsDataset, ToxicCommentDataModule
from lib.model import ToxicCommentTagger


np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)

# 1. Load data.

train = pd.read_csv(Config.DATA_PATH / "train.csv")
test = pd.read_csv(Config.DATA_PATH / "test.csv")
submission = pd.read_csv(Config.DATA_PATH / "submission.csv")

train = train.dropna()
del train['id']

train = train.rename(columns={"toxicity": "label", "comment_text": "text"})
test = test.rename(columns={"comment_text": "text"})

ohe = OneHotEncoder()
expanded_label = ohe.fit_transform(train['label'].values.reshape(-1, 1)).toarray()
del train['label']

for i in range(Config.NUM_LABELS):
    train[f'label_{i}'] = expanded_label[:, i]

# 2. Trasnform data.
X = train["text"].values
y = train["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    train["text"].values, train[[f"label_{i}" for i in range(Config.NUM_LABELS)]].values, test_size=0.05, random_state=42
)


tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
train_dataset = ToxicCommentsDataset(X_train, y_train, tokenizer, max_token_len=Config.MAX_TOKEN_LEN)

# 3. Prepare model.
bert_model = BertModel.from_pretrained(Config.BERT_MODEL_NAME, return_dict=True)

data_module = ToxicCommentDataModule(
    X_train, y_train, X_val, y_val, tokenizer, batch_size=Config.BATCH_SIZE, max_token_len=Config.MAX_TOKEN_LEN
)
data_module.setup()

steps_per_epoch = X_train.shape[0] // Config.BATCH_SIZE
total_training_steps = steps_per_epoch * Config.N_EPOCHS

warmup_steps = total_training_steps // 5
print(f"warmup_steps={warmup_steps}, total_training_steps={total_training_steps}")

model = ToxicCommentTagger(
    n_classes=Config.NUM_LABELS,
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps,
    bert_model_name=Config.BERT_MODEL_NAME,
)

# 4. Train model.

checkpoint_callback = ModelCheckpoint(
    dirpath=Path(Config.DATA_PATH) / "checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)
logger = TensorBoardLogger(Path(Config.DATA_PATH) / "lightning_logs", name="toxic-comments")
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=Config.N_EPOCHS,
    accelerator='gpu',
    devices=1,
)

trainer.fit(model, data_module)

# 5. Make a prediction.

trained_model = ToxicCommentTagger.load_from_checkpoint(
  Path(Config.DATA_PATH) / "checkpoints/best-checkpoint.ckpt",
  n_classes=Config.NUM_LABELS
)
trained_model.eval()
trained_model.freeze()

test_texts = test['text'].values
prediction_column = []

# THIS CODE IS NOT OPTIMISED
for test_comment in tqdm(test_texts):
    encoding = tokenizer.encode_plus(
        test_comment,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    _, test_prediction = trained_model(encoding["input_ids"][:, :512], encoding["attention_mask"][:, :512])
    test_prediction = test_prediction.flatten().numpy()
    label = np.argmax(test_prediction)
    prediction_column.append(label)

test['prediction'] = prediction_column
test[['id', 'prediction']].to_csv(Path(Config.DATA_PATH) / "vanilla_submission.csv", index=False)
