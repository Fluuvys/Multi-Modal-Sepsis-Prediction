import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.nn.functional import kl_div, softmax, log_softmax
import math
from PIL import Image
import torchvision.transforms as transforms


class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv

class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()
        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)
        return loss_rank

class CosineLoss(nn.Module):
    def forward(self, cxr, ehr ):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        return loss


class LSTM(nn.Module):

    def __init__(self, input_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():
            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        out = self.dense_layer(feats)
        scores = torch.sigmoid(out)
        return scores, feats


class CXRModels(nn.Module):

    def __init__(self, args, device='cpu'):
        super(CXRModels, self).__init__()
        self.args = args
        self.device = device
        self.vision_backbone = getattr(torchvision.models, self.args.vision_backbone)(pretrained=self.args.pretrained)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, self.args.vision_num_classes))
        self.feats_dim = d_visual

    def forward(self, x, labels=None, n_crops=0, bs=16):
        lossvalue_bce = torch.zeros(1).to(self.device)
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)
        preds = torch.sigmoid(preds)
        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)
        return preds, lossvalue_bce, visual_feats


class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
            nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        )

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)

    def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
        cxr_preds, _ , feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
            }

    def forward(self, x, seq_lengths=None, img=None, pairs=None ):
        if self.args.fusion_type == 'uni_cxr':
            return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        elif self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type == 'lstm':
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        elif self.args.fusion_type == 'uni_ehr_lstm':
            return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
            }

    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):
        ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)
        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)
        return {
            'early': fused_preds,
            'joint': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
            }

    def forward_lstm_fused(self, x, seq_lengths=None, img=None, pairs=None ):
        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            _, _ , cxr_feats = self.cxr_model(img)
            feats = cxr_feats[:,None,:]
            ehr_feats = self.projection(ehr_feats)
            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            _, _ , cxr_feats = self.cxr_model(img)
            cxr_feats = self.projection(cxr_feats)
            cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs] = 2

        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)
        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        fused_preds = self.lstm_fused_cls(out)

        return {
            'lstm': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
        }

    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None ):
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:,None,:]
        seq_lengths = np.array([1] * len(seq_lengths))
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)
        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        fused_preds = self.lstm_fused_cls(out)
        return {
            'uni_ehr_lstm': fused_preds,
        }


class _MedFuseArgs:
    def __init__(self, fusion_type, vision_backbone, pretrained, num_classes,
                 vision_num_classes, labels_set):
        self.fusion_type = fusion_type
        self.vision_backbone = vision_backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.vision_num_classes = vision_num_classes
        self.labels_set = labels_set


class MedFuseBaseline(nn.Module):

    def __init__(
        self,
        ehr_input_dim: int = 17,
        ehr_hidden_dim: int = 128,
        ehr_layers: int = 1,
        ehr_dropout: float = 0.0,
        vision_backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
        vision_num_classes: int = 14,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.ehr_input_dim = ehr_input_dim
        self.img_size = kwargs.get("img_size", 224)
        self.cxr_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.args = _MedFuseArgs(
            fusion_type="lstm",
            vision_backbone=vision_backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            vision_num_classes=vision_num_classes,
            labels_set="sepsis_onset",
        )

        self.ehr_model = LSTM(
            input_dim=ehr_input_dim,
            num_classes=num_classes,
            hidden_dim=ehr_hidden_dim,
            batch_first=True,
            dropout=ehr_dropout,
            layers=ehr_layers,
        )
        self.cxr_model = CXRModels(self.args, device=device)
        self.fusion = Fusion(self.args, self.ehr_model, self.cxr_model)

        self.to(device)

    def _bin_ts_events(self, ts: dict, t_hours: torch.Tensor):
        hours = ts["hours"].detach().cpu().numpy()
        var_idx = ts["var_idx"].detach().cpu().numpy()
        values = ts["value"].detach().cpu().numpy()
        mask = ts["mask"].detach().cpu().numpy()
        t_hours_np = t_hours.detach().cpu().numpy()

        B, L = hours.shape
        seq_lengths = np.maximum(1, np.ceil(t_hours_np).astype(np.int64))
        T = int(seq_lengths.max())

        x = np.zeros((B, T, self.ehr_input_dim), dtype=np.float32)
        for i in range(B):
            Ti = int(seq_lengths[i])
            for j in range(L):
                if not mask[i, j]:
                    continue
                v_idx = int(var_idx[i, j])
                if v_idx < 0 or v_idx >= self.ehr_input_dim:
                    continue
                bucket = min(max(int(hours[i, j]), 0), Ti - 1)
                x[i, bucket, v_idx] = values[i, j]

        x_t = torch.from_numpy(x)
        seq_lengths_t = torch.as_tensor(seq_lengths, dtype=torch.int64)
        return x_t, seq_lengths_t

    def _load_cxr_images(self, cxr: dict):
        paths = cxr["path"]
        B = len(paths)
        imgs = torch.zeros(B, 3, self.img_size, self.img_size, dtype=torch.float32)
        pairs = np.zeros(B, dtype=bool)
        for i, plist in enumerate(paths):
            if not plist:
                continue
            try:
                image = Image.open(plist[-1]).convert("RGB")
                imgs[i] = self.cxr_transform(image)
                pairs[i] = True
            except (FileNotFoundError, OSError):
                pairs[i] = False
        return imgs, pairs

    def forward(self, batch: dict) -> torch.Tensor:
        ts = batch["ts"]
        cxr = batch["cxr"]
        t_hours = batch["t_hours"]
        batch_size = ts["value"].shape[0]

        x, seq_lengths = self._bin_ts_events(ts, t_hours)
        img, pairs = self._load_cxr_images(cxr)

        x = x.to(self.device)
        img = img.to(self.device)

        out = self.fusion(x, seq_lengths=seq_lengths, img=img, pairs=pairs)
        probs = out["lstm"]
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        probs = probs.view(batch_size, -1)[:, :1]

        eps = 1e-6
        probs = probs.clamp(min=eps, max=1 - eps)
        logits = torch.log(probs) - torch.log1p(-probs)
        return logits.squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, L = 4, 30
    dummy_batch = {
        "t_hours": torch.tensor([20.0, 15.0, 8.0, 20.0]),
        "label": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "ts": {
            "hours": torch.rand(B, L) * 20,
            "var_idx": torch.randint(0, 17, (B, L)),
            "value": torch.randn(B, L),
            "mask": torch.ones(B, L, dtype=torch.bool),
        },
        "cxr": {
            "path": [[], [], [], []],
            "mask": torch.zeros(B, 1, dtype=torch.bool),
        },
    }

    model = MedFuseBaseline(pretrained=False, device="cpu")
    logits = model(dummy_batch)
    print("logits shape:", logits.shape)
    loss = nn.BCEWithLogitsLoss()(logits, dummy_batch["label"])
    loss.backward()
    print("loss:", loss.item())