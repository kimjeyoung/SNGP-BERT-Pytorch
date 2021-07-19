import torch
import os
import numpy as np
from tqdm import tqdm
from models.bert import BertModel, Config
from models.sngp import SNGP, Deterministic
from data_loader import DatasetLoader
from torch.utils.data import DataLoader
from models.optimizers import BertAdam
from utils import to_numpy, Accumulator, mean_field_logits
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc


class Trainer:
    def __init__(self, args):
        t_total = -1
        self.epochs = args.epochs
        self.device = args.device
        self.method = args.method
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.mean_field_factor = args.mean_field_factor
        if args.train_or_test == 'train':
            self.train_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                               max_len=args.max_len, train_or_test='train')
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                           drop_last=True)

            self.val_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                             max_len=args.max_len, train_or_test='val')
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            t_total = len(self.train_loader) * args.epochs

        self.test_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                          max_len=args.max_len, train_or_test='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        # num_classes is number of valid intents plus out-of-scope intent
        self.num_classes = self.test_dataset.num_classes

        # default config is bert-base
        self.bert_config = Config()
        self.backbone = BertModel(self.bert_config)
        self.backbone.load_pretrain_huggingface(torch.load(args.bert_ckpt))
        if args.method == 'sngp':
            self.sngp_model = SNGP(self.backbone,
                                   hidden_size=self.bert_config.hidden_size,
                                   num_classes=self.num_classes,
                                   num_inducing=args.gp_hidden_dim,
                                   n_power_iterations=args.n_power_iterations,
                                   spec_norm_bound=args.spectral_norm_bound,
                                   device="cuda" if self.device == 'gpu' else 'cpu')
        else:
            self.sngp_model = Deterministic(self.backbone,
                                            hidden_size=self.bert_config.hidden_size,
                                            num_classes=self.num_classes)

        if args.device == 'gpu':
            self.sngp_model = self.sngp_model.to("cuda")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = BertAdam(self.sngp_model.parameters(), lr=args.lr,
                                  warmup=args.warmup, weight_decay=args.weight_decay, t_total=t_total)

        if args.train_or_test == 'test' and os.path.isfile(os.path.join(args.save_path, "bestmodel_{}.bin".format(self.method))):
            self.sngp_model.load_state_dict(torch.load(os.path.join(args.save_path, "bestmodel_{}.bin".format(self.method))))

    def train(self):
        best_acc = 0.
        for epoch in range(self.epochs):
            cnt = 0
            self.sngp_model.train()
            metrics = Accumulator()
            loader = tqdm(self.train_loader, disable=False)
            loader.set_description('[%s %04d/%04d]' % ('train', epoch, self.epochs))
            for i, batch in enumerate(loader):
                cnt += self.batch_size
                if self.device == 'gpu':
                    batch = [x.to('cuda') for x in batch]
                self.optimizer.zero_grad()
                x_ids, x_segs, x_attns, label = batch
                pred = self.sngp_model(x_ids, x_segs, x_attns, update_cov=True)
                loss = self.criterion(pred, label)
                acc = accuracy_score(to_numpy(label), to_numpy(torch.argmax(pred, dim=-1))) * 100

                metrics.add_dict({
                    'loss': loss.item() * self.batch_size,
                    'accuracy': acc * self.batch_size,
                })
                postfix = metrics / cnt
                loader.set_postfix(postfix)
                loss.backward()
                self.optimizer.step()

            val_acc = self.eval()
            if val_acc > best_acc:
                best_acc = val_acc
                test_auroc, test_auprc, test_acc = self.test()
                print(f'\t Val dataset --> Best ACC : {best_acc:.3f}')
                print(f'\t Test dataset --> AUROC : {test_auroc:.3f} | AUPRC: {test_auprc:.3f} | ACC: {test_acc:.3f}')
                torch.save(self.sngp_model.state_dict(), os.path.join(self.save_path, "bestmodel_{}.bin".format(self.method)))

            # reset precision matrix
            if self.method == 'sngp':
                self.sngp_model.reset_cov()

    def eval(self):
        self.sngp_model.eval()
        y_true = []
        y_pred = []
        for i, batch in enumerate(self.val_loader):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            self.optimizer.zero_grad()
            x_ids, x_segs, x_attns, label = batch
            pred = self.sngp_model(x_ids, x_segs, x_attns)
            pred = to_numpy(torch.argmax(pred, dim=-1)).flatten().tolist()
            true = to_numpy(label).flatten().tolist()
            y_true.extend(true)
            y_pred.extend(pred)
        acc = accuracy_score(y_true, y_pred) * 100
        return acc

    def test(self):
        self.sngp_model.eval()
        y_true = []
        y_preds = []
        ood_preds = []
        for i, batch in enumerate(tqdm(self.test_loader)):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            self.optimizer.zero_grad()
            x_ids, x_segs, x_attns, label = batch
            logit, cov = self.sngp_model(x_ids, x_segs, x_attns, return_gp_cov=True, update_cov=False)
            if self.method == 'sngp':
                logit = mean_field_logits(logit, cov, mean_field_factor=self.mean_field_factor)
            probs_list = torch.softmax(logit, dim=-1)

            cls_pred = to_numpy(torch.argmax(probs_list, dim=-1)).flatten().tolist()
            ood_pred = to_numpy(1. - torch.max(probs_list, dim=-1)[0]).flatten().tolist()
            true = to_numpy(label).flatten().tolist()
            y_true.extend(true)
            y_preds.extend(cls_pred)
            ood_preds.extend(ood_pred)

        # ood class idx is 150
        ood_true = (np.array(y_true) == 150).astype(np.uint8).tolist()
        test_auroc = roc_auc_score(ood_true, ood_preds)
        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(ood_true, ood_preds)
        test_auprc = auc(recall, precision)

        # calculate accuracy for in-domain
        indomain_true = np.array(y_true)[np.where(np.array(ood_true) == 0)[0]]
        indomain_pred = np.array(y_preds)[np.where(np.array(ood_true) == 0)[0]]
        test_acc = accuracy_score(indomain_true, indomain_pred)
        return test_auroc, test_auprc, test_acc

