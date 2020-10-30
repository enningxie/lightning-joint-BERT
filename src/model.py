# -*- coding: utf-8 -*-
# Created by xieenning at 2020/10/27
from pytorch_lightning import LightningModule
from argparse import Namespace, ArgumentParser
from transformers import BertModel, BertConfig
import torch.nn as nn
import torch
from typing import List, Any
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.metrics import Fbeta


class JointBERT(LightningModule):
    def __init__(self, hparams: Namespace, intent_weight=None):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = hparams['hparams']
        self.save_hyperparameters()
        self.hparams = hparams

        # TODO: 必要超参数
        self.slot_num_labels = hparams.slot_num_labels
        self.intent_num_labels = hparams.intent_num_labels
        self.model_name_or_path = hparams.model_name_or_path

        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.] * self.intent_num_labels)

        self._build_model()
        self._build_loss()
        self._build_metrics()
        self.apply(self._init_weights)

    def _build_model(self):
        self.bert_config = BertConfig.from_pretrained(self.model_name_or_path)
        # output (sequence_output, pooled_output)
        self.bert = BertModel.from_pretrained(self.model_name_or_path, config=self.bert_config)

        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        # intent classification
        self.intent_classifier_head = nn.Linear(self.bert_config.hidden_size, self.intent_num_labels)
        # slot classification
        self.slot_classifier_head = nn.Linear(self.bert_config.hidden_size, self.slot_num_labels)

    def _build_loss(self):
        """ Initializes the loss function/s. """
        self.intent_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = nn.CrossEntropyLoss()

    def _build_metrics(self):
        self.intent_f1_score_train = Fbeta()
        self.slot_f1_score_train = Fbeta(num_classes=self.slot_num_labels)
        self.intent_f1_score_val = Fbeta()
        self.slot_f1_score_val = Fbeta(num_classes=self.slot_num_labels)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output, pooled_output = outputs[0], outputs[1]

        # intent part
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier_head(pooled_output)

        # slot part
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier_head(sequence_output)

        return intent_logits, slot_logits

    def _calculate_loss(self, batch):
        input_ids, attention_mask, token_type_ids, slot_mask, slot_label, intent_label = batch

        # inference
        intent_logits, slot_logits = self.forward(input_ids, attention_mask, token_type_ids)

        # intent part loss
        intent_loss = self.intent_loss_fct(intent_logits, intent_label.type_as(intent_logits))

        # slot part loss
        active_loss = slot_mask.view(-1) == 1
        active_logits = slot_logits.view(-1, self.slot_num_labels)
        active_labels = torch.where(
            active_loss, slot_label.view(-1), torch.tensor(self.slot_loss_fct.ignore_index).type_as(slot_label)
        )
        slot_loss = self.slot_loss_fct(active_logits, active_labels)
        return intent_logits, slot_logits, intent_loss, slot_loss

    def training_step(self, batch, batch_idx):
        _, _, _, slot_mask, slot_label, intent_label = batch
        intent_logits, slot_logits, intent_loss, slot_loss = self._calculate_loss(batch)

        self.log('intent_loss', intent_loss)
        self.log('slot_loss', slot_loss)

        intent_pred = (intent_logits > 0).to(dtype=torch.long)  # (b_z, intent_dim)

        _, slot_pred = slot_logits.max(dim=2)  # (b_z, max_length)

        slot_mask = slot_mask.view(-1) == 1
        slot_pred_ = torch.masked_select(slot_pred.view(-1), slot_mask)
        slot_label_ = torch.masked_select(slot_label.view(-1), slot_mask)

        # val intent F1
        self.intent_f1_score_train(intent_pred, intent_label)
        # val slot F1
        self.slot_f1_score_train(slot_pred_, slot_label_)

        self.log("intent_f1", self.intent_f1_score_train, on_step=True, prog_bar=True)
        self.log("slot_f1", self.slot_f1_score_train, on_step=True, prog_bar=True)

        return intent_loss + slot_loss

    def validation_step(self, batch, batch_idx):
        _, _, _, slot_mask, slot_label, intent_label = batch
        intent_logits, slot_logits, intent_loss, slot_loss = self._calculate_loss(batch)

        self.log('val_intent_loss', intent_loss)
        self.log('val_slot_loss', slot_loss)

        intent_pred = (intent_logits > 0).to(dtype=torch.long)  # (b_z, intent_dim)

        _, slot_pred = slot_logits.max(dim=2)  # (b_z, max_length)

        slot_mask = slot_mask.view(-1) == 1
        slot_pred_ = torch.masked_select(slot_pred.view(-1), slot_mask)
        slot_label_ = torch.masked_select(slot_label.view(-1), slot_mask)

        # val intent F1
        self.intent_f1_score_val(intent_pred, intent_label)
        # val slot F1
        self.slot_f1_score_val(slot_pred_, slot_label_)

        self.log("intent_f1_val", self.intent_f1_score_val, on_step=True, on_epoch=True)
        self.log("slot_f1_val", self.slot_f1_score_val, on_step=True, on_epoch=True)

    def validation_epoch_end(
            self, outputs: List[Any]
    ) -> None:
        intent_f1_val_epoch = self.intent_f1_score_val.compute()
        slot_f1_val_epoch = self.slot_f1_score_val.compute()
        self.log("intent_f1_val_epoch", intent_f1_val_epoch, on_epoch=True)
        self.log("slot_f1_val_epoch", slot_f1_val_epoch, on_epoch=True)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if
                        not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
        #                                             num_training_steps=self.hparams.max_step)
        return [optimizer], []

    @classmethod
    def add_model_specific_args(
            cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        # "/Data/public/pretrained_models/pytorch/chinese-bert-wwm-ext"
        # "/Data/public/pretrained_models/chinese-electra-180g-base-discriminator"
        parser.add_argument(
            "--model_name_or_path",
            default="/Data/public/pretrained_models/pytorch/chinese-bert-wwm-ext",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        # parser.add_argument("--num_labels", default=2, type=int)
        return parser
