"""
SPEC++ Fine-Tuning and Evaluation
Silas Ayitey, Kwabena Owusu-Agyemang
Kwame Nkrumah University of Science and Technology
"""

import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip

DATA_ROOT = '/content/SPEC_local_data'
SUBSETS = ['absolute_size', 'relative_size', 'absolute_spatial',
           'relative_spatial', 'existence', 'count']


class SPECDataset(Dataset):
    def __init__(self, data_root, subset, tokenizer, preprocess):
        self.subset_dir = os.path.join(data_root, subset)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        with open(os.path.join(self.subset_dir, 'image2text.json')) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.subset_dir, sample['query'])
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        text_tokens = self.tokenizer(sample['keys'])
        return {
            'image': image,
            'texts': text_tokens,
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }


class SplitSPECDataset(SPECDataset):
    def __init__(self, data_root, subset, tokenizer, preprocess,
                 split='train', train_ratio=0.8):
        self.subset_dir = os.path.join(data_root, subset)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        with open(os.path.join(self.subset_dir, 'image2text.json')) as f:
            all_samples = json.load(f)
        n_train = int(len(all_samples) * train_ratio)
        if split == 'train':
            self.samples = all_samples[:n_train]
        else:
            self.samples = all_samples[n_train:]


def spec_collate(batch):
    images = torch.stack([b['image'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    texts_list = [b['texts'] for b in batch]
    K_max = max(t.shape[0] for t in texts_list)
    L = texts_list[0].shape[1]
    padded, mask = [], []
    for t in texts_list:
        K_i = t.shape[0]
        if K_i < K_max:
            t_padded = torch.cat([t, torch.zeros(K_max - K_i, L, dtype=t.dtype)], dim=0)
        else:
            t_padded = t
        padded.append(t_padded)
        m = torch.zeros(K_max, dtype=torch.bool)
        m[:K_i] = True
        mask.append(m)
    return {
        'image': images,
        'texts': torch.stack(padded),
        'label': labels,
        'mask': torch.stack(mask),
    }


class FixedTrainer:
    def __init__(self, model, optimizer, device, temperature=100.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature

    def train_step(self, batch):
        self.model.train()
        images = batch['image'].to(self.device)
        texts = batch['texts'].to(self.device)
        labels = batch['label'].to(self.device)
        mask = batch['mask'].to(self.device)
        B, K_max, L = texts.shape

        img_feat = F.normalize(self.model.encode_image(images), dim=-1)
        txt_feat = self.model.encode_text(texts.view(B * K_max, L))
        txt_feat = F.normalize(txt_feat, dim=-1).view(B, K_max, -1)

        sim = torch.einsum('bd,bkd->bk', img_feat, txt_feat)
        logits = sim * self.temperature
        logits = logits.masked_fill(~mask, float('-inf'))

        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


class AdaptiveHardNegativeTrainer:
    def __init__(self, model, optimizer, device, temperature=100.0, alpha=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, batch):
        self.model.train()
        images = batch['image'].to(self.device)
        texts = batch['texts'].to(self.device)
        labels = batch['label'].to(self.device)
        mask = batch['mask'].to(self.device)
        B, K_max, L = texts.shape

        img_feat = F.normalize(self.model.encode_image(images), dim=-1)
        txt_feat = self.model.encode_text(texts.view(B * K_max, L))
        txt_feat = F.normalize(txt_feat, dim=-1).view(B, K_max, -1)

        sim = torch.einsum('bd,bkd->bk', img_feat, txt_feat)
        logits = sim * self.temperature
        logits = logits.masked_fill(~mask, float('-inf'))

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            correct_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            confusion = 1.0 - correct_prob

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        weights = 1.0 + self.alpha * confusion
        weights = weights / weights.mean()
        loss = (ce_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


@torch.no_grad()
def evaluate_subset(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=spec_collate, num_workers=0)
    correct, total = 0, 0
    for batch in loader:
        images = batch['image'].to(device)
        texts = batch['texts'].to(device)
        labels = batch['label'].to(device)
        mask = batch['mask'].to(device)
        B, K_max, L = texts.shape
        img_feat = F.normalize(model.encode_image(images), dim=-1)
        txt_feat = model.encode_text(texts.view(B * K_max, L))
        txt_feat = F.normalize(txt_feat, dim=-1).view(B, K_max, -1)
        similarity = torch.einsum('bd,bkd->bk', img_feat, txt_feat)
        similarity = similarity.masked_fill(~mask, float('-inf'))
        preds = similarity.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / total if total > 0 else 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device).train()

    train_datasets, test_datasets = {}, {}
    for subset in SUBSETS:
        if os.path.isdir(os.path.join(DATA_ROOT, subset)):
            train_datasets[subset] = SplitSPECDataset(
                DATA_ROOT, subset, tokenizer, preprocess, split='train')
            test_datasets[subset] = SplitSPECDataset(
                DATA_ROOT, subset, tokenizer, preprocess, split='test')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    trainer = AdaptiveHardNegativeTrainer(model, optimizer, device, alpha=0.1)

    for subset_name, train_ds in train_datasets.items():
        print(f"Training on {subset_name}")
        loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                            collate_fn=spec_collate, num_workers=0)
        step = 0
        while step < 300:
            for batch in loader:
                if step >= 300:
                    break
                loss = trainer.train_step(batch)
                step += 1

    results = {}
    for subset_name, test_ds in test_datasets.items():
        acc = evaluate_subset(model, test_ds, device)
        results[subset_name] = acc
        print(f"  {subset_name}: {acc:.2%}")


if __name__ == '__main__':
    main()
