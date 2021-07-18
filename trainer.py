import torch
import os
from models.bert import BertModel, Config
from models.sngp import SNGP
from data_loader import DatasetLoader
from torch.utils.data import DataLoader
from models.optimizers import BertAdam


class Trainer:
    def __init__(self, args):
        t_total = -1
        if args.train_or_test == 'train':
            self.train_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                               max_len=args.max_len, train_or_test='train')
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                           drop_last=True)

            self.val_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                             max_len=args.max_len, train_or_test='val')
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            t_total = len(self.train_loader)

        self.test_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                          max_len=args.max_len, train_or_test='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        self.num_classes = self.test_dataset.num_classes

        # default config is bert-base
        self.bert_config = Config()
        self.backbone = BertModel(self.bert_config, checkpoint=args.bert_ckpt)
        self.sngp_model = SNGP(self.backbone,
                               hidden_size=self.bert_config.hidden_size,
                               num_classes=self.num_classes,
                               num_inducing=args.gp_hidden_dim,
                               n_power_iterations=args.n_power_iterations,
                               spec_norm_bound=args.spectral_norm_bound)
        if args.device == 'gpu':
            self.sngp_model = self.sngp_model.to("cuda")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = BertAdam(self.sngp_model.parameters(), lr=args.lr,
                                  warmup=args.warmup, weight_decay=args.weight_decay, t_total=t_total)

        if args.train_or_test == 'test' and os.path.isfile(os.path.join(args.save_path, "bestmodel.bin")):
            self.sngp_model.load_state_dict(torch.load(os.path.join(args.save_path, "bestmodel.bin")))

    def train(self):
        print("TEST")

    def test(self):
        print("TSET")