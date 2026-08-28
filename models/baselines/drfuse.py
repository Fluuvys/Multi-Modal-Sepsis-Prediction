from __future__ import annotations
"""
drfuse.py -- baseline reproduction of DrFuse (AAAI 2024), adapted for binary
Sepsis-3 onset prediction. Adapted into experiments/train.py's MODEL_REGISTRY,
same pattern as models/baselines/utde.py in this project.

SOURCE: Yao et al., "DrFuse: Learning Disentangled Representation for Clinical
Multi-Modal Fusion with Missing Modality and Modal Inconsistency." AAAI 2024.
Code: github.com/dorothy-yao/drfuse (drfuse.py, ehr_transformer.py,
drfuse_trainer.py, as pasted into this project's chat history -- no LICENSE
file was present in that repo at the time this was vendored; keep this
attribution if you redistribute anything beyond internal baseline
reproduction, per PROJECT_CONTEXT.md rule #1's "independently reproduced"
requirement, same caveat noted in utde.py).

WHAT THIS FILE DOES relative to the three vendored files:
  * Merges DrFuseModel (drfuse.py) and EHRTransformer (ehr_transformer.py)
    into one file, per task requirement #1.
  * Head change: the vendored model took `num_classes=len(label_names)` (25
    ICD-derived phenotype labels in the original CXR-benchmark setup). Every
    shape in DrFuseModel's disease-aware attention block (attn_proj's
    `(2+num_classes)*hidden_size` fan-out, the per-class attention logits,
    the `torch.diagonal` trick in the final head) was already written
    generically over `num_classes` -- it happens to degenerate cleanly to
    `num_classes=1` for binary Sepsis-3, so no structural changes were needed
    there, only removing the final `.sigmoid()` calls (see next point).
  * BCELoss(reduction='none') on already-sigmoid'd branch predictions ->
    every branch (`pred_final`, `pred_ehr`, `pred_cxr`, `pred_shared`) now
    returns a raw LOGIT, and all loss terms use BCEWithLogitsLoss /
    binary_cross_entropy_with_logits instead. The JSD disentanglement term is
    untouched by this -- it was never computed on classification outputs, it
    operates on `.sigmoid()` of the *projected shared features themselves*
    (`feat_ehr_shared` / `feat_cxr_shared`), which has nothing to do with the
    prediction head and needed no change for the binary task.
  * All multilabel_average_precision / multilabel_auroc epoch loops from
    drfuse_trainer.py's on_validation_epoch_end / on_test_epoch_end (looped
    over 25 disease labels) are stripped entirely -- this project's
    evaluate.py owns AUROC/AUPRC/ECE uniformly across every baseline
    (PROJECT_CONTEXT.md rule #1), a model file has no business computing its
    own eval metrics.
  * pl.LightningModule (training_step/validation_step/configure_optimizers)
    is replaced by a plain nn.Module ADAPTER (`DrFuseBaseline`, bottom of
    this file) matching experiments/train.py's ACTUAL MODEL_REGISTRY contract
    -- confirmed by reading train.py's `run_epoch`, not guessed:
    `forward(batch) -> logits [B]` (a bare tensor). `run_epoch` does
    `logits = model(batch); loss = loss_fn(logits, batch["label"])` directly
    on whatever `forward` returns, and also does `torch.sigmoid(logits)` on
    it -- a dict return breaks both immediately. An earlier version of this
    file returned `{"pred_final": ..., "loss_aux": ...}`, which was wrong;
    fixed below.
  * KNOWN LIMITATION from that same fix: train.py's training loop has no
    mechanism today for adding an auxiliary loss on top of the primary BCE
    (MODEL_REGISTRY entries are plain `(config, device) -> nn.Module`
    builders with a fixed `forward(batch) -> logits` contract). DrFuseLosses'
    disentanglement / branch-prediction / attention-ranking terms are still
    computed every forward pass (for potential logging / a future train.py
    change) and stashed on `self.last_loss_aux` / `self.last_aux_components`,
    but are NOT currently part of the backward pass -- only the primary
    BCEWithLogitsLoss on `pred_final`, computed by train.py itself, trains
    this model right now. If you want the auxiliary losses to actually do
    something, `run_epoch` needs a small patch, e.g.:
        logits = model(batch)
        loss = loss_fn(logits, batch["label"])
        if hasattr(model, "last_loss_aux"):
            loss = loss + config.get("lambda_aux", 1.0) * model.last_loss_aux
    which would also generalize to SDCA/SARL's own auxiliary losses later.

CAVEAT -- same class of limitation as models/baselines/medfuse.py, explicitly
flagged per the task issue: DrFuse's disentangled shared/distinct design is
built for exactly two modalities (one "distinct" stream per side, one
pairwise shared-alignment term via JSD). It does not extend to a third
modality (notes) without redesigning attn_proj's fan-out and the JSD pairing
logic for a 3-way shared space. **This baseline is TS + CXR only -- notes are
dropped entirely, not degraded gracefully.** Do not use this architecture as
a stepping stone toward 3-modality fusion; models/ours/backbone.py is the
from-scratch fusion architecture built to scale past two modalities.
"""


import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

# ==========================================
# 1. BASE MODULES (vendored from ehr_transformer.py, unchanged)
# ==========================================

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


class EHRTransformer(nn.Module):
    """Vendored as-is from ehr_transformer.py, EXCEPT the final `.sigmoid()` on
    `pred_distinct` is removed -- this branch head now returns a raw logit
    (see module docstring, "BCELoss ... -> BCEWithLogitsLoss")."""

    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        x = self.emb(x)
        x = self.pos_encoder(x)
        feat = self.model_feat(x, src_key_padding_mask=attn_mask)
        h_shared = self.model_shared(feat, src_key_padding_mask=attn_mask)
        h_distinct = self.model_distinct(feat, src_key_padding_mask=attn_mask)

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask == float('-inf')] = 0
        rep_shared = (padding_mask * h_shared).sum(dim=1) / padding_mask.sum(dim=1)
        rep_distinct = (padding_mask * h_distinct).sum(dim=1) / padding_mask.sum(dim=1)

        pred_distinct = self.fc_distinct(rep_distinct)  # logit now, was .sigmoid()

        return rep_shared, rep_distinct, pred_distinct


# ==========================================
# 2. MAIN MODEL (vendored from drfuse.py, head made binary)
# ==========================================

class DrFuseModel(nn.Module):
    """Vendored from drfuse.py with two changes, both confined to where each
    branch head's output leaves the sigmoid, per the module docstring:
      - pred_cxr / pred_shared / pred_final no longer call `.sigmoid()` --
        every returned prediction is a raw logit.
      - `num_classes` is expected to be 1 (binary Sepsis-3) by the adapter at
        the bottom of this file, but nothing in this class hardcodes that --
        it is left generic exactly as vendored, since the disease-aware
        attention math already falls out correctly for num_classes=1.
    """

    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head,
                 ehr_input_size=76, cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(input_size=ehr_input_size, num_classes=num_classes,
                                        d_model=hidden_size, n_head=ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                        n_layers_distinct=ehr_n_layers,
                                        dropout=ehr_dropout)

        resnet = resnet50()
        self.cxr_model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        resnet = resnet50()
        self.cxr_model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_shared.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        resnet = resnet50()
        self.cxr_model_spec = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )
        self.cxr_model_spec.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_size)

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.cxr_model_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.attn_proj = nn.Linear(hidden_size, (2+num_classes)*hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, img, seq_lengths, pairs, grl_lambda):
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model_feat(img)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)

        # get shared feature
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct)  # logit, was .sigmoid()

        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)

        pairs = pairs.unsqueeze(1)

        h1 = feat_ehr_shared
        h2 = feat_cxr_shared
        term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)

        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared)  # logit, was .sigmoid()

        # Disease-wise (here: single-class) attention
        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1-pairs) * q[:, :-1].mean(dim=1)

        # compute attention weighting
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze()==0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)  # logit, was followed by .sigmoid()

        outputs = {
            'feat_ehr_shared': feat_ehr_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
        }

        return outputs


# ==========================================
# 3. LOSSES (vendored from drfuse_trainer.py, delighted of pl.LightningModule
#    and of the metric-logging that came bundled with every loss method.
#    Every masked-pred-loss helper now takes logits + BCEWithLogitsLoss
#    instead of sigmoid outputs + BCELoss -- see module docstring.)
# ==========================================

class JSD(nn.Module):
    """Vendored unchanged -- operates on .sigmoid()'d *shared features*, not
    on classification outputs, so the binary-task head change doesn't touch it."""

    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())


class DrFuseLosses(nn.Module):
    """Bundles every loss term from DrFuseTrainer's `_compute_and_log_loss` /
    `_disentangle_loss_jsd` / `_compute_prediction_losses`, minus all
    `self.log_dict(...)` calls (no Lightning logger here -- experiments/train.py
    decides what to log) and minus the primary `loss_pred_final` term, which
    the adapter deliberately leaves for train.py to compute uniformly across
    every baseline (see `DrFuseBaseline.forward`'s docstring for why).

    lambda_* weights are read from a plain namespace/dict (`hparams`) instead
    of `self.hparams` (that was populated by Lightning's `save_hyperparameters`).
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams_ = hparams
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='none')  # was BCELoss on sigmoid outputs
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.jsd = JSD()

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs):
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        jsd = self.jsd(model_output['feat_ehr_shared'].sigmoid(),
                       model_output['feat_cxr_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams_['lambda_disentangle_shared'] * jsd +
                                self.hparams_['lambda_disentangle_ehr'] * loss_sim_ehr +
                                self.hparams_['lambda_disentangle_cxr'] * loss_sim_cxr)
        return loss_disentanglement, {'EHR_distinct': loss_sim_ehr.detach(),
                                       'CXR_distinct': loss_sim_cxr.detach(),
                                       'shared_jsd': jsd.detach()}

    def _compute_branch_prediction_losses(self, model_output, y_gt, pairs):
        # NOTE: unlike the vendored version, this does NOT include loss_pred_final
        # -- that's the primary loss, owned by experiments/train.py, applied
        # identically across every baseline (PROJECT_CONTEXT.md rule #1).
        ehr_mask = torch.ones_like(model_output['pred_final'][:, 0])
        loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
        loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)
        return loss_pred_ehr, loss_pred_cxr, loss_pred_shared

    def _attn_ranking_loss(self, model_output, y_gt, pairs):
        raw_pred_loss_ehr = F.binary_cross_entropy_with_logits(model_output['pred_ehr'], y_gt, reduction='none')
        raw_pred_loss_cxr = F.binary_cross_entropy_with_logits(model_output['pred_cxr'], y_gt, reduction='none')
        raw_pred_loss_shared = F.binary_cross_entropy_with_logits(model_output['pred_shared'], y_gt, reduction='none')

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output['attn_weights']
        attn_ehr, attn_shared, attn_cxr = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2]

        cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
        loss_attn1 = pairs * F.margin_ranking_loss(attn_cxr, attn_ehr, cxr_overweights_ehr, reduction='none')
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1 > 0].numel())

        shared_overweights_ehr = 2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
        loss_attn2 = pairs * F.margin_ranking_loss(attn_shared, attn_ehr, shared_overweights_ehr, reduction='none')
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2 > 0].numel())

        shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
        loss_attn3 = pairs * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3 > 0].numel())

        return (loss_attn1 + loss_attn2 + loss_attn3) / 3

    def forward(self, model_output, y_gt, pairs):
        """Returns (loss_aux, components_dict). y_gt is expected shape [B, 1]
        (binary target, float) to line up with pred_* shape [B, 1]."""
        loss_pred_ehr, loss_pred_cxr, loss_pred_shared = self._compute_branch_prediction_losses(
            model_output, y_gt, pairs)
        loss_branch_pred = (self.hparams_['lambda_pred_shared'] * loss_pred_shared +
                            self.hparams_['lambda_pred_ehr'] * loss_pred_ehr +
                            self.hparams_['lambda_pred_cxr'] * loss_pred_cxr)

        loss_disentanglement, disentangle_components = self._disentangle_loss_jsd(model_output, pairs)
        loss_attn_ranking = self._attn_ranking_loss(model_output, y_gt, pairs)

        loss_aux = (loss_branch_pred + loss_disentanglement +
                   self.hparams_['lambda_attn_aux'] * loss_attn_ranking)

        components = {
            'pred_ehr': loss_pred_ehr.detach(),
            'pred_cxr': loss_pred_cxr.detach(),
            'pred_shared': loss_pred_shared.detach(),
            'attn_ranking': loss_attn_ranking.detach(),
            **disentangle_components,
        }
        return loss_aux, components




# ==========================================
# ADAPTER -- bridges the vendored TS+CXR-only, 2-modality model above to this
# project's ACTUAL batch schema (confirmed via inspect_batch.py against real
# `collate_sepsis_batch` output, not guessed) and MODEL_REGISTRY contract.
#
# CONFIRMED SCHEMA (inspect_batch.py output):
#   batch["t_hours"]:            [B]         float32 -- this timepoint's own
#                                                        hour-since-admission
#   batch["label"]:              [B]         float32
#   batch["ts"]["hours"]:        [B, T_raw]  float32 -- per-EVENT
#                                                        hour-since-admission
#   batch["ts"]["var_idx"]:      [B, T_raw]  int64
#   batch["ts"]["value"]:        [B, T_raw]  float32
#   batch["ts"]["mask"]:         [B, T_raw]  bool
#   batch["cxr"]["hours"]:       [B, T_cxr]  float32  (T_cxr observed == 1)
#   batch["cxr"]["path"]:        list[B] of ragged lists of file-path strings
#                                             (NOT preloaded image tensors)
#   batch["cxr"]["mask"]:        [B, T_cxr]  bool
#
# FIX (previous version of this file, before train.py/dataset.py were seen):
#   1. `forward` returned a dict; train.py's `run_epoch` needs a bare `[B]`
#      logits tensor (`loss_fn(logits, ...)`, `torch.sigmoid(logits)` both
#      break on a dict). Fixed -- see `DrFuseBaseline.forward` below and the
#      module docstring's KNOWN LIMITATION on where this leaves the
#      auxiliary losses.
#   2. Assumed a per-event `hours_before_onset` field that doesn't exist.
#      There IS a per-event `ts["hours"]` (confirmed above), but it's
#      hour-since-ADMISSION, not hour-before-onset or hour-before-this-
#      timepoint. Recency for binning has to be computed as
#      `age = t_hours - ts["hours"]` using the newly-confirmed top-level
#      `t_hours` field. Fixed in `_discretize_ts` below.
#   3. Assumed CXR was a padded sequence of preloaded image tensors needing
#      a "most recent" selection. Actually: `T_cxr` is already 1 (SepsisDataset
#      already reduces CXR history to a single most-recent candidate per
#      timepoint -- no selection needed here), and `cxr["path"]` holds file
#      PATHS, not tensors -- images have to be loaded from disk and
#      transformed here. Fixed in `_load_cxr_batch` below.
#
# STILL OPEN / worth double-checking against dataset.py directly:
#   - Whether `cxr["path"][b]` can ever contain more than one path (T_cxr
#     was 1 in the inspected batch, but this code defensively takes the LAST
#     path in the list rather than assuming exactly one).
#   - The lead-time sweep mechanism. train.py's `SepsisDataset(...)` call in
#     `main()` takes no `lead_hours`/`horizon` argument at all -- so
#     `model_args.lead_hours` in a config yaml (if you had one) currently
#     does NOTHING; nothing in this adapter or in train.py reads it. Either
#     the lead time is baked into `sepsis_labels.parquet` itself (e.g. one
#     parquet file per lead time, swapped via `data_dir`), or into
#     `obs_buffer_hours`, or SepsisDataset has a constructor argument train.py
#     doesn't expose yet. Check dataset.py's `SepsisDataset.__init__` for how
#     the 2h/4h/6h/12h sweep is actually parameterized, and drop any
#     `lead_hours` key from configs/drfuse.yaml until you confirm it does
#     something -- keeping a dead config key around invites someone assuming
#     a sweep is happening when it isn't.
# ==========================================

DEFAULT_MODEL_ARGS = dict(
    hidden_size=128,
    ehr_n_head=4,
    ehr_n_layers=1,
    ehr_dropout=0.3,
    lookback_hours=48,        # number of hourly bins fed to EHRTransformer,
                               # counted backward from each sample's own
                               # t_hours. ASSUMPTION: chosen to match the
                               # original MIMIC-III-benchmark-style 48h
                               # window DrFuse's paper used, not retuned for
                               # sepsis lead times.
    cxr_image_size=224,        # resnet50 input side length
    # ASSUMPTION: the four lambda_disentangle_* / lambda_pred_* /
    # lambda_attn_aux weights below were tuned in the original DrFuse repo
    # for a 25-label ICD-phenotype multilabel task on MIMIC-IV+CXR. Nothing
    # in this project has re-tuned them for binary Sepsis-3 -- left at 1.0
    # (i.e. "no term dominates by construction") as a neutral starting point.
    # Currently unused for backprop until train.py is patched (see module
    # docstring's KNOWN LIMITATION) -- kept here so that patch has real
    # weights to reach for.
    lambda_pred_ehr=1.0,
    lambda_pred_cxr=1.0,
    lambda_pred_shared=1.0,
    lambda_disentangle_shared=1.0,
    lambda_disentangle_ehr=1.0,
    lambda_disentangle_cxr=1.0,
    lambda_attn_aux=1.0,
)


def _discretize_ts(ts: dict, t_hours: torch.Tensor, lookback_hours: int, n_vars: int, device):
    """Turns the ragged, admission-absolute-timestamped
    (hours, var_idx, value, mask) event stream into the dense
    [B, lookback_hours, 2*n_vars] grid EHRTransformer expects (value channel
    + observed-mask channel per variable -- the standard
    MIMIC-III-benchmark-style discretization DrFuse's original 76-dim input
    assumed, 76 = 2 * 38 there, 2 * n_vars here).

    `ts['hours']` is hour-since-admission for each event (confirmed via
    inspect_batch.py); `t_hours` is this timepoint's OWN hour-since-admission
    (top-level batch field, confirmed present). `age = t_hours - ts['hours']`
    is therefore "how many hours before this prediction point did this event
    happen" -- 0 = just happened, larger = further in the past. This is
    floored into an hourly bin and placed oldest-to-newest (bin
    `lookback_hours - 1` = the most recent hour), matching
    LearnablePositionalEncoding's assumption of chronological order.

    Events with `age < 0` (an event timestamped AFTER this prediction point
    -- shouldn't happen if SepsisDataset windows correctly, kept as a
    defensive drop rather than an assertion so one bad row doesn't crash a
    whole epoch) or `age >= lookback_hours` (older than the lookback window)
    are dropped. Multiple events landing in the same (sample, hour,
    variable) cell are mean-aggregated; empty cells are forward-filled from
    the last observed value for that variable (falling back to 0 if there's
    no prior observation at all).

    Returns (x, seq_lengths) where x is [B, lookback_hours, 2*n_vars] and
    seq_lengths is currently a constant `[lookback_hours] * B` -- ASSUMPTION:
    every sample uses the full lookback window (hours with no true history
    forward-fill from 0 with mask=0, which the model can learn to discount);
    a tighter version would derive true admission length to shorten this per
    sample, which would need an additional field this adapter doesn't use
    today.
    """
    value = ts['value'].to(device)
    var_idx = ts['var_idx'].to(device)
    mask = ts['mask'].to(device).bool()
    hours = ts['hours'].to(device)
    t_hours = t_hours.to(device).unsqueeze(1)  # [B, 1], broadcasts against [B, T_raw]

    B, T_raw = value.shape
    age = t_hours - hours  # [B, T_raw]; 0 = just happened, larger = further in the past
    bin_from_end = age.floor().long()
    keep = mask & (bin_from_end >= 0) & (bin_from_end < lookback_hours)
    bin_idx = (lookback_hours - 1 - bin_from_end).clamp(0, lookback_hours - 1)
    var_idx_c = var_idx.clamp(min=0, max=n_vars - 1)

    val_sum = torch.zeros(B, lookback_hours, n_vars, device=device)
    val_cnt = torch.zeros(B, lookback_hours, n_vars, device=device)
    for b in range(B):
        idx = keep[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        bins_b = bin_idx[b, idx]
        vars_b = var_idx_c[b, idx]
        vals_b = value[b, idx]
        val_sum[b].index_put_((bins_b, vars_b), vals_b, accumulate=True)
        val_cnt[b].index_put_((bins_b, vars_b), torch.ones_like(vals_b), accumulate=True)

    observed = val_cnt > 0
    binned_value = torch.where(observed, val_sum / val_cnt.clamp(min=1), torch.zeros_like(val_sum))

    filled_value = binned_value.clone()
    for t in range(1, lookback_hours):
        carry = ~observed[:, t]
        filled_value[:, t][carry] = filled_value[:, t - 1][carry]

    x = torch.cat([filled_value, observed.float()], dim=-1)  # [B, T, 2*n_vars]
    seq_lengths = [lookback_hours] * B

    return x, seq_lengths


_CXR_TRANSFORM_CACHE = {}


def _get_cxr_transform(image_size: int):
    """Standard ImageNet-style preprocessing for the resnet50 backbone.
    ASSUMPTION: `cxr_model_feat`'s resnet50() is randomly initialized (not
    pretrained -- see DrFuseModel, vendored as-is), so ImageNet normalization
    stats matter less than with pretrained weights, but are kept anyway
    since resizing/tensor-conversion is required regardless and there's no
    reason to skip standard normalization for a randomly-initialized net."""
    if image_size not in _CXR_TRANSFORM_CACHE:
        from torchvision import transforms
        _CXR_TRANSFORM_CACHE[image_size] = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _CXR_TRANSFORM_CACHE[image_size]


def _load_cxr_batch(cxr_batch: dict, image_size: int, device):
    """Loads CXR images from disk paths -- `cxr_batch['path']` is a ragged
    list of per-sample path lists (confirmed via inspect_batch.py: NOT a
    preloaded image tensor). `T_cxr` (the tensor dim on `mask`/`hours`) was
    observed to already be 1 -- SepsisDataset appears to hand back at most
    one "most recent" CXR candidate per timepoint already -- but this
    defensively takes the LAST path in a sample's list in case that's ever
    not true, rather than assuming exactly one.

    A sample with no CXR available (empty path list) gets a zeroed image and
    `pairs[b] == 0`, which is exactly the signal DrFuseModel's own `pairs`
    gating expects for "this modality is absent for this sample" -- the CXR
    branch still runs on the batch (dense ops, same as the vendored code),
    it just gets masked out of the fusion/loss downstream.
    """
    paths = cxr_batch['path']
    B = len(paths)
    transform = _get_cxr_transform(image_size)

    imgs = []
    pairs = torch.zeros(B)
    for b in range(B):
        sample_paths = paths[b]
        loaded = False
        if sample_paths:
            path = sample_paths[-1]
            try:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                imgs.append(transform(img))
                pairs[b] = 1.0
                loaded = True
            except (FileNotFoundError, OSError) as e:
                # ASSUMPTION: an unreadable/missing file is treated the same
                # as "no CXR for this sample" rather than crashing the whole
                # epoch over one bad path. Confirm this is the behavior you
                # want -- a silently-skipped corrupt file could also be worth
                # surfacing as a data-quality issue upstream.
                print(f"WARNING: could not load CXR at {path!r} ({e}); treating as missing.")
        if not loaded:
            imgs.append(torch.zeros(3, image_size, image_size))

    img_batch = torch.stack(imgs, dim=0).to(device)
    return img_batch, pairs.to(device)


class DrFuseBaseline(nn.Module):
    """MODEL_REGISTRY contract (confirmed against train.py's `run_epoch`):
    `forward(batch) -> logits [B]`, a bare tensor -- see the module
    docstring's FIX note for why this changed from a dict.

    Auxiliary losses (branch prediction losses, JSD disentanglement,
    attention ranking) are computed every forward pass and stashed on
    `self.last_loss_aux` (scalar tensor, NOT detached) / `self.last_aux_components`
    (dict, for logging), but are NOT part of the backward pass under train.py
    as it stands today -- see the module docstring's KNOWN LIMITATION for the
    small train.py patch that would change that. Only `label` present in
    `batch` triggers this computation at all; `self.last_loss_aux` is a
    `0.`-valued, grad-free tensor otherwise (e.g. pure inference with no
    labels).

    Scope reminder (see module docstring's CAVEAT): TS + CXR only. If
    `config["modalities"]` requests notes, this raises rather than silently
    dropping the modality, since DrFuse's disentanglement math has nowhere to
    put a third stream.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        modalities = config.get("modalities", ["ts", "cxr"])
        if "notes" in modalities:
            raise ValueError(
                "DrFuseBaseline is TS+CXR only (see this file's module "
                "docstring CAVEAT) -- 'notes' in config['modalities'] is not "
                "supported. Use models/ours/backbone.py for 3-modality runs."
            )
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        self.lookback_hours = int(model_args['lookback_hours'])
        self.cxr_image_size = int(model_args['cxr_image_size'])
        self.device_ = device

        try:
            from dataset import VARIABLE_VOCAB
            self.n_ts_vars = len(VARIABLE_VOCAB)
        except ImportError:
            # ASSUMPTION: fallback vocab size for standalone execution / the
            # __main__ smoke test below, where `dataset` isn't importable
            # (e.g. running this file directly rather than via train.py,
            # which puts experiments/ on sys.path). Real training must
            # resolve the import above.
            self.n_ts_vars = 34

        self.model = DrFuseModel(
            hidden_size=model_args['hidden_size'],
            num_classes=1,
            ehr_dropout=model_args['ehr_dropout'],
            ehr_n_head=model_args['ehr_n_head'],
            ehr_n_layers=model_args['ehr_n_layers'],
            ehr_input_size=2 * self.n_ts_vars,
        )
        self.losses = DrFuseLosses(model_args)
        self.last_loss_aux = torch.zeros(())
        self.last_aux_components = {}

    def forward(self, batch: dict) -> torch.Tensor:
        x, seq_lengths = _discretize_ts(
            batch["ts"], batch["t_hours"], self.lookback_hours, self.n_ts_vars, self.device_)
        img, pairs = _load_cxr_batch(batch["cxr"], self.cxr_image_size, self.device_)

        out = self.model(x, img, seq_lengths, pairs, grl_lambda=0)
        pred_final = out['pred_final'].squeeze(-1)  # [B, 1] -> [B]

        if "label" in batch:
            y_gt = batch["label"].float().to(self.device_).view(-1, 1)
            loss_aux, components = self.losses(out, y_gt, pairs)
            self.last_loss_aux = loss_aux  # NOT detached -- a train.py patch
                                            # can call .backward() through this
            self.last_aux_components = components
        else:
            self.last_loss_aux = torch.zeros((), device=self.device_)
            self.last_aux_components = {}

        return pred_final


if __name__ == '__main__':
    # Definition-of-done smoke test: instantiate the model, generate mock
    # TS + CXR tensors, run a forward pass end-to-end (both the raw
    # DrFuseModel directly, matching the original repo's own sanity checks,
    # and through the project adapter with a mock batch dict matching the
    # SCHEMA CONFIRMED via inspect_batch.py), and assert dimensional
    # alignment. No real data or `dataset.py` import required; CXR paths
    # point at temp files this block creates itself so image loading is
    # actually exercised, not just shape-checked.
    import tempfile
    from PIL import Image

    torch.manual_seed(0)
    B, T, F_IN, H = 4, 48, 76, 32

    print("--- 1. Raw DrFuseModel forward (vendored architecture, no adapter) ---")
    model = DrFuseModel(hidden_size=H, num_classes=1, ehr_dropout=0.1, ehr_n_layers=1, ehr_n_head=4,
                        ehr_input_size=F_IN)
    x = torch.randn(B, T, F_IN)
    img = torch.randn(B, 3, 224, 224)
    seq_lengths = [T] * B
    pairs = torch.tensor([1., 1., 0., 1.])
    out = model(x, img, seq_lengths, pairs, grl_lambda=0)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert out['pred_final'].shape == (B, 1), "pred_final should be [B, 1] logits before the adapter squeezes it"

    print("\n--- 2. DrFuseLosses (binary BCEWithLogits variant) ---")
    losses = DrFuseLosses(DEFAULT_MODEL_ARGS)
    y_gt = torch.randint(0, 2, (B, 1)).float()
    loss_aux, components = losses(out, y_gt, pairs)
    print(f"  loss_aux: {loss_aux.item():.4f}")
    print(f"  components: { {k: round(v.item(), 4) for k, v in components.items()} }")
    assert torch.isfinite(loss_aux), "loss_aux should not be NaN/inf on random init"

    print("\n--- 3. DrFuseBaseline adapter forward (mock batch matching CONFIRMED schema) ---")
    n_vars = 34  # matches the ImportError fallback above when `dataset` isn't on sys.path
    mock_config = {"modalities": ["ts", "cxr"], "model_args": {"hidden_size": H, "lookback_hours": T}}
    adapter = DrFuseBaseline(mock_config, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        cxr_path = f"{tmpdir}/mock_cxr.png"
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(cxr_path)

        T_raw = 200  # raw irregular event count per sample, before discretization
        admission_hour_now = 60.0  # each sample's t_hours -- events must be <= this
        mock_batch = {
            "t_hours": torch.full((B,), admission_hour_now),
            "ts": {
                "value": torch.randn(B, T_raw),
                "var_idx": torch.randint(0, n_vars, (B, T_raw)),
                "mask": torch.ones(B, T_raw),
                # hour-since-admission per event, spread over the last ~60h,
                # some older than the lookback_hours=T=48 window on purpose
                # to exercise the drop-too-old path
                "hours": torch.rand(B, T_raw) * admission_hour_now,
            },
            "cxr": {
                "hours": torch.tensor([[55.], [50.], [0.], [58.]]),
                "path": [[cxr_path], [cxr_path], [], [cxr_path]],  # sample 2 has no CXR
                "mask": torch.tensor([[True], [True], [False], [True]]),
            },
            "label": torch.tensor([1., 0., 1., 0.]),
        }
        adapter_out = adapter(mock_batch)

    print(f"  logits: {tuple(adapter_out.shape)} (expect ({B},))")
    print(f"  last_loss_aux: {adapter.last_loss_aux.item():.4f}")
    assert adapter_out.shape == (B,)
    assert torch.isfinite(adapter.last_loss_aux)

    # Mirrors run_epoch's exact usage in train.py
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(adapter_out, mock_batch["label"])
    probs = torch.sigmoid(adapter_out)
    print(f"  train.py-style loss: {loss.item():.4f} | probs: {probs.tolist()}")
    print("\nAll shape/finiteness checks passed.")
