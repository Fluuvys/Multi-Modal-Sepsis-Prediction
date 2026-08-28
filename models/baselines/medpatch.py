"""medpatch.py -- baseline reproduction of MedPatch (Al Jorf & Shamout, MLHC 2025),
adapted for binary Sepsis-3 onset prediction. Designed to slot into
experiments/train.py's MODEL_REGISTRY the same way models/baselines/drfuse.py does.

SOURCE: Al Jorf & Shamout, "MedPatch: A Framework for Clinical Multi-Modal Fusion
of Heterogeneous Modalities via Confidence-Based Patching." MLHC 2025.
Code reference: github.com/nyuad-cai/MedPatch (MedPatch/models/fusion.py -- the
CMSMAFusion class -- and MedPatch/models/ehr_models.py). No LICENSE file was
present in that repo at the time of vendoring; keep this attribution if you
redistribute anything beyond internal baseline reproduction, per
PROJECT_CONTEXT.md rule #1's "independently reproduced" requirement, same
caveat noted in drfuse.py and utde.py.

WHAT WE REPRODUCE (the actual MedPatch mechanism):
    * Token-level confidence prediction via an auxiliary per-token head +
      temperature.
    * High/low confidence token partitioning per modality.
    * Per-partition masked-mean pooling -> concat -> linear head.
    * Missingness-aware multi-stage late fusion across modalities with
      learnable stage weights renormalized over present stages.

WHAT WE ADAPT (rather than fork):
    * Encoders: lightweight from-scratch (LSTM for TS, hashing+transformer for
      notes, ViT-tiny for CXR). Reference uses 3-stage pretrained encoders
      with their own data extraction; we start from random init and reuse our
      cohort's preprocessing pipeline (PROJECT_CONTEXT.md rule #5).
    * Calibration: heuristic (per-modality temperature scalar + sigmoid
      binary branch `conf = max(p, 1-p)`), no separate Stage-2 calibration
      pass. Flagged as a deviation in the ASSUMPTIONS section below.
    * Head: raw logits (no final sigmoid) so train.py's BCEWithLogitsLoss is
      numerically correct (same fix drfuse.py:28-34 had to make).
    * Fuser: simple concat+linear with missingness masking, the reference's
      `fuser=None` path (MedPatch/models/fusion.py:1671-1674).
    * Windowing: discretize EHR to hourly bins, take only the most-recent
      few notes per timepoint, take only the most-recent CXR per timepoint.
      These reductions live here as adapters, NOT in preprocessing/, per
      PROJECT_CONTEXT.md rule #5 and docs/data_schema.md:58-59.

WHAT WE DROP (intentionally):
    * 3-stage pretraining schedule (Stage-1 unimodal / Stage-2 confidence
      calibration / Stage-3 joint).
    * 25-label ICU-phenotype multitask head.
    * `args`-namespace config (`arguments.py`) -- we consume a dict from
      `experiments/configs/medpatch.yaml` like every other baseline here.
    * `DataFusion.py` extraction pipeline.

ASSUMPTIONS (per PROJECT_CONTEXT.md rule #8, captured here):
    1. EHR discretization matches drfuse.py's `_discretize_ts` exactly (same
       schema in, same dense [B, L, 2*n_vars] out, same forward-fill
       semantics). Ported verbatim for consistency with the existing
       baseline rather than imported, matching how drfuse.py and utde.py each
       carry their own copy.
    2. CXR load is the same ragged-path -> [B, 3, H, W] loader drfuse.py
       uses, with the same "no CXR for this sample => zeroed image +
       pairs=0" missingness signal. Ported verbatim.
    3. Token-level confidence uses `conf = max(p, 1-p)` (binary branch),
       matching fusion.py:1423. Per-modality temperature scalar initialized
       to 1.0, clamp_min(1e-9), matching fusion.py:1413.
    4. Notes use a deterministic hashing tokenizer (fixed seed,
       `% vocab_size`) so the model is fully self-contained -- no download
       of a pretrained tokenizer, no NLTK/spaCy dependency. Multiple notes
       per timepoint are concatenated most-recent-first up to
       `max_note_tokens`. Medfuse/MedPatch's most-recent-only behavior
       is reproduced via `_select_recent_notes` here.
    5. The MEDPATCH paper assumes paired EHR+CXR+RR+DN at exactly one 48h
       snapshot per admission (mortality task). Our setup is hourly rolling
       prediction with a 4h horizon and explicit missingness handling. The
       architecture (confidence + multi-stage fusion) is task-agnostic, so
       the adaptation is local: windowing on the EHR side, the
       `_select_recent_notes` / `_load_cxr_batch` reductions, and the
       missingness mask that already lives at the bottom of fusion.py.

CAVEAT -- CXR NOT YET LINKED:
    `cxr_metadata.parquet` has 0 rows in both
    `Multi-Modal-Sepsis-Prediction/data/cohort/` and `multimodal_sepsis/Data/cohort/`
    (CXR linking has not finished). The model file is written tri-modal; the
    default config requests only `[ts, notes]`. Turning CXR on is a one-line
    change in `experiments/configs/medpatch.yaml` (uncomment the `cxr` entry
    under `modalities`). When CXR is requested but no paths are present,
    `_load_cxr_batch` returns all-zero images with `pairs=0` and the
    missingness mask zeros out the CXR stage at fusion time -- the model
    degrades to `[ts, notes]` automatically, no crash.

NOT-WIRED NOTE:
    This file is shape-compatible with `train.py`'s MODEL_REGISTRY --
    `MedPatchBaseline(config, device) -> nn.Module` builder, `forward(batch) ->
    logits [B]` bare tensor contract, `last_loss_aux` / `last_aux_components`
    stashed. It is NOT yet registered. The wiring is intentionally left as a
    separate, additive change (3 lines in `experiments/train.py`: import +
    builder + entry in MODEL_REGISTRY) so this file can be reviewed on its
    own. Until wired, running `train.py --config
    experiments/configs/medpatch.yaml` raises `ValueError: Unknown model
    'medpatch'` from `train.py:238`. That is expected and is not a defect
    in this file.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# DEFAULT MODEL ARGS -- mirrors the relevant subset of
# MedPatch/arguments.py so the config file is the single source of truth.
# ============================================================

DEFAULT_MODEL_ARGS = dict(
    hidden_size=128,
    lookback_hours=48,
    max_notes=4,
    max_note_tokens=512,
    note_vocab_size=20000,
    cxr_image_size=224,
    cxr_patch_size=16,
    ts_confidence_threshold=0.3,
    cxr_confidence_threshold=0.3,
    notes_confidence_threshold=0.3,
    weight_high=1.0,
    weight_low=1.0,
    weight_late=1.0,
    ablation=None,  # one of: None, "without_joint_module", "without_missingness_module", "without_late_module"
)


# ============================================================
# ADAPTERS -- the reduction layer between our cohort's master
# Parquet schema and this model's expected inputs.
# ============================================================

def _discretize_ts(
    ts: dict, t_hours: torch.Tensor, lookback_hours: int, n_vars: int, device
):
    """Ported verbatim from models/baselines/drfuse.py:498-568.

    Turns the ragged event stream (hours, var_idx, value, mask) into the
    dense [B, lookback_hours, 2*n_vars] grid this model's TSEncoder
    consumes (value channel + observed-mask channel per variable). See
    drfuse.py's docstring for the full rationale.
    """
    value = ts['value'].to(device)
    var_idx = ts['var_idx'].to(device)
    mask = ts['mask'].to(device).bool()
    hours = ts['hours'].to(device)
    t_hours = t_hours.to(device).unsqueeze(1)  # [B, 1]

    B, T_raw = value.shape
    age = t_hours - hours  # [B, T_raw]; 0 = just happened
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


def _select_recent_notes(
    notes_batch: dict, max_notes: int
) -> Tuple[List[List[str]], torch.Tensor]:
    """Reproduces MedPatch's most-recent-only behavior on notes: for each
    sample, take the `max_notes` most-recent notes (by `hours` -- larger
    means closer to the prediction point, since `hours` is
    hour-since-admission and the prediction point is later in admission
    time), most-recent-first, and return a [B] boolean tensor that's
    True iff at least one note was available.

    SepsisDataset already windows notes to be at-or-before the prediction
    point (`obs_buffer_hours`); the only filter we add is "most recent N."
    Empty notes (empty string after stripping) are dropped.
    """
    paths_lists = notes_batch['path'] if 'path' in notes_batch else None
    hours_lists = notes_batch['hours'] if 'hours' in notes_batch else None
    text_lists = notes_batch['text']  # confirmed schema: ragged list[list[str]]
    mask = notes_batch['mask']  # [B, T_notes]

    B = len(text_lists)
    out_text: List[List[str]] = []
    present = torch.zeros(B, dtype=torch.bool)

    for b in range(B):
        candidates = []
        # NOTE: mask is [B, T_notes] and tells us which slots are real
        # (not padding). Indices into text_lists[b] correspond to the same
        # slots. hours_lists[b] (if present) gives us the ordering key.
        sample_text = text_lists[b]
        sample_mask = mask[b] if mask.dim() == 2 else None
        if hours_lists is not None:
            sample_hours = hours_lists[b]
        else:
            sample_hours = [0.0] * len(sample_text)

        for i, t in enumerate(sample_text):
            if sample_mask is not None and not bool(sample_mask[i]):
                continue
            cleaned = t.strip() if isinstance(t, str) else ""
            if not cleaned:
                continue
            h = float(sample_hours[i]) if i < len(sample_hours) else 0.0
            candidates.append((h, cleaned))
        # most-recent first
        candidates.sort(key=lambda x: x[0], reverse=True)
        kept = [c[1] for c in candidates[:max_notes]]
        out_text.append(kept)
        present[b] = len(kept) > 0
    return out_text, present


_CXR_TRANSFORM_CACHE = {}


def _get_cxr_transform(image_size: int):
    """Standard ImageNet preprocessing for the CXR encoder (ViT-tiny, random
    init). Ported from drfuse.py:574-588 with no changes -- the preprocessing
    is correct regardless of backbone, and resnet50/ViT-tiny both expect
    [0, 1] -> normalized tensors.
    """
    if image_size not in _CXR_TRANSFORM_CACHE:
        from torchvision import transforms
        _CXR_TRANSFORM_CACHE[image_size] = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _CXR_TRANSFORM_CACHE[image_size]


def _load_cxr_batch(cxr_batch: dict, image_size: int, device):
    """Ported verbatim from models/baselines/drfuse.py:591-634.

    Loads CXR images from disk paths (ragged list of per-sample path lists),
    defensively taking the LAST path in a sample's list. A sample with no
    CXR available gets a zeroed image and `pairs[b] == 0`, which the
    missingness mask then zeros out at fusion time.
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
                print(f"WARNING: could not load CXR at {path!r} ({e}); treating as missing.")
        if not loaded:
            imgs.append(torch.zeros(3, image_size, image_size))

    img_batch = torch.stack(imgs, dim=0).to(device)
    return img_batch, pairs.to(device)


# ============================================================
# ENCODERS -- each returns (pooled_feats [B, D], per_token [B, L, D]).
# The two-output shape is what the confidence mechanism needs
# (MedPatch/models/ehr_models.py:67-71 returns the same under 'c-' in
# fusion_type).
# ============================================================

class TSEncoder(nn.Module):
    """LSTM over hourly-binned, 2*n_vars-dim EHR vectors. Mirrors
    MedPatch/models/ehr_models.py:9-73 (the LSTM path, not the transformer
    variant). Pack-padded-sequence is wired in even though our discretizer
    returns the full lookback window today; the contract lets us tighten
    `seq_lengths` later without changing this module.
    """

    def __init__(self, n_vars: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(2 * n_vars, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, seq_lengths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, 2*n_vars]
        h = self.proj(x)
        # Pack so the LSTM ignores padding properly.
        lengths = torch.tensor(seq_lengths, dtype=torch.long, device=h.device)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h_n, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        pooled = h_n.squeeze(0)  # [B, D]
        return pooled, out  # [B, D], [B, L, D]


class NotesEncoder(nn.Module):
    """Hashing tokenizer + small transformer. Deterministic, dependency-free,
    and the design choice the user confirmed: no pretrained tokenizer
    download, no NLTK/spaCy.

    Tokenization: lowercase -> non-alphanumerics -> whitespace split ->
    SHA-1 -> first 8 hex chars -> int -> % vocab_size. Stable across runs.

    Multiple notes per sample are concatenated most-recent-first up to
    `max_note_tokens` total tokens. Padded with a learned PAD token at
    index 0; token_mask marks real tokens.
    """

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, vocab_size: int, hidden_size: int, max_tokens: int, depth: int = 2, heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        # index 0 is reserved for PAD
        self.emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=hidden_size * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    @classmethod
    def _hash_token(cls, tok: str, vocab_size: int) -> int:
        h = hashlib.sha1(tok.encode("utf-8")).hexdigest()
        return (int(h[:8], 16) % (vocab_size - 1)) + 1  # 0 reserved for PAD

    @classmethod
    def tokenize(cls, text: str, vocab_size: int) -> List[int]:
        if not text:
            return []
        toks = cls._WORD_RE.findall(text.lower())
        return [cls._hash_token(t, vocab_size) for t in toks]

    def forward(self, text_lists: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build a [B, max_tokens] token-id batch, most-recent-first per sample.
        B = len(text_lists)
        ids = torch.zeros(B, self.max_tokens, dtype=torch.long)
        token_mask = torch.zeros(B, self.max_tokens, dtype=torch.bool)

        for b, notes in enumerate(text_lists):
            cursor = 0
            for note in notes:
                if cursor >= self.max_tokens:
                    break
                toks = self.tokenize(note, self.vocab_size)
                room = self.max_tokens - cursor
                toks = toks[:room]
                if toks:
                    ids[b, cursor:cursor + len(toks)] = torch.tensor(toks, dtype=torch.long)
                    token_mask[b, cursor:cursor + len(toks)] = True
                    cursor += len(toks)

        h = self.emb(ids) + self.pos[:, :self.max_tokens]
        out = self.transformer(h)  # [B, L_tok, D]
        # Mean-pool over real tokens only.
        denom = token_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (out * token_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return pooled, out  # [B, D], [B, L_tok, D]


class CXREncoder(nn.Module):
    """ViT-tiny: Conv2d patchify -> learned positional -> 2-layer
    transformer -> CLS-pooled + per-patch features. Random init. Shape
    numbers come from `image_size=224, patch_size=16` -> 14x14 = 196 patches
    + 1 CLS = 197 tokens; with `patch_size=16` we use 224/16=14.
    """

    def __init__(self, image_size: int, patch_size: int, hidden_size: int, depth: int = 2, heads: int = 4):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=hidden_size * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # img: [B, 3, H, W]
        B = img.size(0)
        patches = self.proj(img)  # [B, D, H/p, W/p]
        patches = patches.flatten(2).transpose(1, 2)  # [B, n_patches, D]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, patches], dim=1) + self.pos[:, :self.n_patches + 1]
        out = self.transformer(h)  # [B, n_patches+1, D]
        pooled = out[:, 0]  # CLS
        return pooled, out  # [B, D], [B, L, D]


# ============================================================
# CONFIDENCE PREDICTOR -- per-token sigmoid head + temperature.
# ============================================================

class ConfidencePredictor(nn.Module):
    """Linear(D, 1) per modality -> sigmoid -> `conf = max(p, 1-p)`
    (the binary branch, fusion.py:1423). Per-modality learned temperature
    scalar, clamp_min(1e-9) (fusion.py:1413).

    Heuristic by design: no separate Stage-2 calibration pass. The user's
    confirmed choice -- trade exact-paper-faithfulness for a self-contained
    training pipeline that fits the rest of this project's baselines.
    """

    def __init__(self, hidden_size: int, init_temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-9)

    def forward(self, per_token: torch.Tensor) -> torch.Tensor:
        # per_token: [B, L, D] -> confidence [B, L]
        logits = self.proj(per_token).squeeze(-1) / self.temperature()
        p = torch.sigmoid(logits)
        conf = torch.where(p > 1 - p, p, 1 - p)
        return conf


# ============================================================
# CORE MEDPATCH OP: high/low confidence token partitioning.
# ============================================================

def _confidence_partition(
    per_token: torch.Tensor,
    conf: torch.Tensor,
    token_mask: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits per-token features into high/low confidence groups, masked-mean
    pools each, and returns the pooled vectors plus per-partition mean
    confidences.

    `mask_high = conf >= threshold`, `mask_low = ~mask_high`, both AND-ed
    with the real `token_mask` so padding enters neither partition.
    Counts use clamp_min(1e-9), per fusion.py:1448-1451.

    Returns (pooled_high, pooled_low, conf_high, conf_low). Each is [B, D]
    or [B].
    """
    mask_high = (conf >= threshold) & token_mask
    mask_low = (conf < threshold) & token_mask

    def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], m: [B, L] -> [B, D]
        denom = m.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
        return (x * m.unsqueeze(-1).float()).sum(dim=1) / denom

    pooled_high = masked_mean(per_token, mask_high)
    pooled_low = masked_mean(per_token, mask_low)
    conf_high = (conf * mask_high.float()).sum(dim=1) / mask_high.float().sum(dim=1).clamp(min=1e-9)
    conf_low = (conf * mask_low.float()).sum(dim=1) / mask_low.float().sum(dim=1).clamp(min=1e-9)
    return pooled_high, pooled_low, conf_high, conf_low


# ============================================================
# MULTI-STAGE LATE FUSION (the missingness-aware piece).
# ============================================================

class MedPatchFusion(nn.Module):
    """Three prediction stages, matching MedPatch/models/fusion.py:
        1. Per-modality unimodal logits (LayerNorm + Linear on the pooled
           encoder features).
        2. High-confidence joint logit: concat per-modality high pools ->
           linear head. The reference's `fuser=None` path (fusion.py:1671-1674).
        3. Low-confidence joint logit (same shape, low pools).
        4. Missingness-only logit: Linear(n_modalities, 1) on the binary
           present/absent vector (fusion.py:1190, 1708).
        5. Late fusion: softmax-normalized learnable weights [n_mods + 3]
           over the extended missingness mask, renormalized over present
           stages, weighted sum over component logits (fusion.py:1737-1773).

    Ablations (selected via `config["model_args"]["ablation"]`, reusing
    reference names so Table 3 rows line up):
        * "without_joint_module": zero out the high/low stages, only
          unimodal + missingness contribute.
        * "without_missingness_module": drop the missingness stage.
        * "without_late_module": average components with equal weights
          instead of softmax-renormalized learnable weights.

    All outputs are RAW LOGITS (no sigmoid anywhere) -- same correction
    drfuse.py:28-34 had to make, so train.py's BCEWithLogitsLoss is
    numerically correct.
    """

    ABLATIONS = (
        None,
        "without_joint_module",
        "without_missingness_module",
        "without_late_module",
    )

    def __init__(self, n_modalities: int, hidden_size: int, ablation: Optional[str] = None):
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {self.ABLATIONS}"
            )
        self.ablation = ablation
        self.n_modalities = n_modalities

        self.unimodal_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
            for _ in range(n_modalities)
        ])
        self.high_head = nn.Sequential(
            nn.LayerNorm(n_modalities * hidden_size),
            nn.Linear(n_modalities * hidden_size, 1),
        )
        self.low_head = nn.Sequential(
            nn.LayerNorm(n_modalities * hidden_size),
            nn.Linear(n_modalities * hidden_size, 1),
        )
        self.missingness_head = nn.Linear(n_modalities, 1)

        # Learnable stage weights: one per modality + high + low + missingness.
        n_stages = n_modalities + 3
        self.stage_weights = nn.Parameter(torch.zeros(n_stages))
        # Initialize so softmax produces roughly equal weights.
        nn.init.normal_(self.stage_weights, mean=0.0, std=0.01)

    def forward(
        self,
        pooled_per_modality: List[torch.Tensor],
        high_pool: torch.Tensor,
        low_pool: torch.Tensor,
        present_mask: torch.Tensor,  # [B, n_modalities], bool
    ) -> dict:
        B = present_mask.size(0)
        present_mask_f = present_mask.float()

        # 1. Unimodal logits, [B, n_modalities]
        # 1. Unimodal logits, [B, n_modalities]
        # NOTE: each head(p) is [B, 1]; concatenate along dim=1 to get [B, n_modalities]
        # directly. Squeezing per-head to [B] before cat (the old code) collapses the
        # batch dim away, so cat(dim=-1) flattens everything into [B * n_modalities]
        # instead of stacking modalities as columns -- that's what produced the
        # "size 2 vs 128" broadcast error against uni_w_present ([B, n_modalities]).
        uni = torch.cat([
            head(p) for head, p in zip(self.unimodal_heads, pooled_per_modality)
        ], dim=1)

        # 2-4. Joint + missingness logits
        high_logit = self.high_head(high_pool).squeeze(-1)
        low_logit = self.low_head(low_pool).squeeze(-1)
        miss_logit = self.missingness_head(present_mask_f).squeeze(-1)

        # 5. Late fusion weights.
        if self.ablation == "without_joint_module":
            n_stages = self.n_modalities + 1  # uni + missingness, no high/low
            weights = torch.softmax(self.stage_weights[:n_stages], dim=0)
            # No high/low => high_logit, low_logit unused.
            uni_w = weights[:self.n_modalities]
            miss_w = weights[self.n_modalities]
            uni_mask = present_mask_f
            uni_w_present = uni_w.unsqueeze(0) * uni_mask
            uni_w_present = uni_w_present / uni_w_present.sum(dim=1, keepdim=True).clamp(min=1e-9)
            late_logit = (uni_w_present * uni).sum(dim=1) + miss_w * miss_logit
        elif self.ablation == "without_missingness_module":
            weights = torch.softmax(self.stage_weights[:-1], dim=0)  # drop the missingness slot
            uni_w = weights[:self.n_modalities]
            high_w = weights[self.n_modalities]
            low_w = weights[self.n_modalities + 1]
            uni_mask = present_mask_f
            uni_w_present = uni_w.unsqueeze(0) * uni_mask
            uni_w_present = uni_w_present / uni_w_present.sum(dim=1, keepdim=True).clamp(min=1e-9)
            late_logit = (
                (uni_w_present * uni).sum(dim=1)
                + high_w * high_logit
                + low_w * low_logit
            )
        elif self.ablation == "without_late_module":
            # Equal weights, still per-modality-masked for the uni branches.
            n_stages = self.n_modalities + 3
            weights = torch.full((n_stages,), 1.0 / n_stages, device=uni.device)
            uni_w = weights[:self.n_modalities]
            high_w = weights[self.n_modalities]
            low_w = weights[self.n_modalities + 1]
            miss_w = weights[self.n_modalities + 2]
            uni_mask = present_mask_f
            uni_w_present = uni_w.unsqueeze(0) * uni_mask
            uni_w_present = uni_w_present / uni_w_present.sum(dim=1, keepdim=True).clamp(min=1e-9)
            late_logit = (
                (uni_w_present * uni).sum(dim=1)
                + high_w * high_logit
                + low_w * low_logit
                + miss_w * miss_logit
            )
        else:
            weights = torch.softmax(self.stage_weights, dim=0)
            uni_w = weights[:self.n_modalities]
            high_w = weights[self.n_modalities]
            low_w = weights[self.n_modalities + 1]
            miss_w = weights[self.n_modalities + 2]
            uni_mask = present_mask_f
            uni_w_present = uni_w.unsqueeze(0) * uni_mask
            uni_w_present = uni_w_present / uni_w_present.sum(dim=1, keepdim=True).clamp(min=1e-9)
            late_logit = (
                (uni_w_present * uni).sum(dim=1)
                + high_w * high_logit
                + low_w * low_logit
                + miss_w * miss_logit
            )

        return {
            "uni": uni,                  # [B, n_modalities]
            "high_logit": high_logit,    # [B]
            "low_logit": low_logit,      # [B]
            "miss_logit": miss_logit,    # [B]
            "late_logit": late_logit,    # [B]
        }


# ============================================================
# REGISTRY-SHAPED ADAPTER.
# ============================================================

class MedPatchBaseline(nn.Module):
    """MODEL_REGISTRY contract (matches drfuse.py's adapter):
        * `__init__(config, device)` -- (config, device) -> nn.Module builder.
        * `forward(batch) -> logits [B]` -- bare tensor (run_epoch does
          `loss_fn(logits, batch["label"])` and `torch.sigmoid(logits)`
          directly).
        * `self.last_loss_aux` / `self.last_aux_components` stashed on
          every forward with labels (same docstring'd limitation as
          drfuse.py: auxiliary losses are NOT part of the backward pass
          under today's train.py; we don't fix that here, and the
          comment above the class notes the small train.py patch that
          would).
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        modalities = list(config.get("modalities", ["ts", "notes"]))
        self.modalities = modalities
        self.device_ = device
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        self.lookback_hours = int(model_args["lookback_hours"])
        self.cxr_image_size = int(model_args["cxr_image_size"])
        self.cxr_patch_size = int(model_args["cxr_patch_size"])
        self.hidden_size = int(model_args["hidden_size"])
        self.max_notes = int(model_args["max_notes"])
        self.max_note_tokens = int(model_args["max_note_tokens"])
        self.note_vocab_size = int(model_args["note_vocab_size"])
        self.weight_high = float(model_args["weight_high"])
        self.weight_low = float(model_args["weight_low"])
        self.weight_late = float(model_args["weight_late"])
        self.ts_conf_threshold = float(model_args["ts_confidence_threshold"])
        self.cxr_conf_threshold = float(model_args["cxr_confidence_threshold"])
        self.notes_conf_threshold = float(model_args["notes_confidence_threshold"])
        self.ablation = model_args.get("ablation", None)

        # n_ts_vars from the project vocab (17 in production). Fallback for
        # standalone smoke-test runs where `dataset` isn't importable.
        try:
            from dataset import VARIABLE_VOCAB  # noqa: F401
            self.n_ts_vars = len(VARIABLE_VOCAB)
        except ImportError:
            self.n_ts_vars = 17

        # Build encoders for whichever modalities are requested.
        if "ts" in modalities:
            self.ts_encoder = TSEncoder(self.n_ts_vars, self.hidden_size)
            self.ts_conf = ConfidencePredictor(self.hidden_size)
            self.thresholds = {"ts": self.ts_conf_threshold}
        if "notes" in modalities:
            self.notes_encoder = NotesEncoder(
                vocab_size=self.note_vocab_size,
                hidden_size=self.hidden_size,
                max_tokens=self.max_note_tokens,
            )
            self.notes_conf = ConfidencePredictor(self.hidden_size)
            self.thresholds = getattr(self, "thresholds", {})
            self.thresholds["notes"] = self.notes_conf_threshold
        if "cxr" in modalities:
            self.cxr_encoder = CXREncoder(
                image_size=self.cxr_image_size,
                patch_size=self.cxr_patch_size,
                hidden_size=self.hidden_size,
            )
            self.cxr_conf = ConfidencePredictor(self.hidden_size)
            self.thresholds = getattr(self, "thresholds", {})
            self.thresholds["cxr"] = self.cxr_conf_threshold

        # Track token_mask per modality for the partition op (so padding
        # tokens never enter either high or low buckets).
        self._token_masks = {}

        # Build fusion once we know how many modalities we actually have.
        self.fusion = MedPatchFusion(
            n_modalities=len(modalities),
            hidden_size=self.hidden_size,
            ablation=self.ablation,
        )

        self.last_loss_aux = torch.zeros(())
        self.last_aux_components = {}

    # ----- per-modality forward helpers -----

    def _forward_ts(self, batch: dict):
        x, seq_lengths = _discretize_ts(
            batch["ts"], batch["t_hours"], self.lookback_hours, self.n_ts_vars, self.device_
        )
        pooled, per_tok = self.ts_encoder(x, seq_lengths)
        # All L timesteps are "real" after discretization (forward-filled).
        token_mask = torch.ones(
            per_tok.size(0), per_tok.size(1), dtype=torch.bool, device=per_tok.device
        )
        assert pooled.dim() == 2 and pooled.size(1) == self.hidden_size, (
            f"TSEncoder returned pooled shape {tuple(pooled.shape)}, "
            f"expected [B, {self.hidden_size}]"
        )
        return pooled, per_tok, token_mask

    def _forward_notes(self, batch: dict):
        text_lists, present = _select_recent_notes(batch["notes"], self.max_notes)
        pooled, per_tok = self.notes_encoder(text_lists)
        # Build token_mask on the SAME device as per_tok (the previous
        # CPU-build version raised device-mismatch errors in training).
        # We use the embedding's padding_idx=0 convention: a slot is real
        # iff its id is non-zero. Re-tokenize is wasteful but keeps the
        # mask definitionally tied to what the encoder actually saw.
        B = per_tok.size(0)
        token_ids = torch.zeros(B, self.max_note_tokens, dtype=torch.long, device=per_tok.device)
        for b, notes in enumerate(text_lists):
            cursor = 0
            for note in notes:
                if cursor >= self.max_note_tokens:
                    break
                toks = NotesEncoder.tokenize(note, self.note_vocab_size)
                room = self.max_note_tokens - cursor
                toks = toks[:room]
                if toks:
                    token_ids[b, cursor:cursor + len(toks)] = torch.tensor(toks, dtype=torch.long, device=per_tok.device)
                    cursor += len(toks)
        token_mask = token_ids != 0
        # Sanity: pooled must be [B, hidden_size]. If a future encoder
        # refactor changes the shape this surfaces the bug at the
        # boundary instead of letting it crash later inside the fusion
        # heads with a confusing "[B, 2] vs [B, 128]" error.
        assert pooled.dim() == 2 and pooled.size(1) == self.hidden_size, (
            f"NotesEncoder returned pooled shape {tuple(pooled.shape)}, "
            f"expected [B, {self.hidden_size}]"
        )
        return pooled, per_tok, token_mask, present

    def _forward_cxr(self, batch: dict):
        img, pairs = _load_cxr_batch(batch["cxr"], self.cxr_image_size, self.device_)
        pooled, per_tok = self.cxr_encoder(img)
        # All patches are real for any image (even the all-zero missingness
        # placeholder -- the partition still picks a confidence value, but
        # pairs=0 in `present_mask` zeroes this modality's unimodal logit
        # contribution at fusion time).
        token_mask = torch.ones(
            per_tok.size(0), per_tok.size(1), dtype=torch.bool, device=per_tok.device
        )
        assert pooled.dim() == 2 and pooled.size(1) == self.hidden_size, (
            f"CXREncoder returned pooled shape {tuple(pooled.shape)}, "
            f"expected [B, {self.hidden_size}]"
        )
        return pooled, per_tok, token_mask, pairs.bool()

    # ----- main forward -----

    def forward(self, batch: dict) -> torch.Tensor:
        pooled_per_modality: List[torch.Tensor] = []
        per_token_per_modality: List[torch.Tensor] = []
        token_masks: List[torch.Tensor] = []
        present_list: List[torch.Tensor] = []

        if "ts" in self.modalities:
            pooled, per_tok, mask = self._forward_ts(batch)
            pooled_per_modality.append(pooled)
            per_token_per_modality.append(per_tok)
            token_masks.append(mask)
            present_list.append(torch.ones(pooled.size(0), dtype=torch.bool, device=pooled.device))

        if "notes" in self.modalities:
            pooled, per_tok, mask, present = self._forward_notes(batch)
            pooled_per_modality.append(pooled)
            per_token_per_modality.append(per_tok)
            token_masks.append(mask)
            present_list.append(present.to(pooled.device))

        if "cxr" in self.modalities:
            pooled, per_tok, mask, present = self._forward_cxr(batch)
            pooled_per_modality.append(pooled)
            per_token_per_modality.append(per_tok)
            token_masks.append(mask)
            present_list.append(present.to(pooled.device))

        # High/low pools: per-modality, then concat.
        high_pools, low_pools, high_confs, low_confs = [], [], [], []
        # Map modality name -> its ConfidencePredictor, in the order
        # self.modalities was declared. Only modalities that were enabled
        # at __init__ time are looked up here (the dict-comprehension
        # version crashed with AttributeError when CXR wasn't built).
        confidences = []
        for m in self.modalities:
            if m == "ts":
                confidences.append(self.ts_conf)
            elif m == "notes":
                confidences.append(self.notes_conf)
            elif m == "cxr":
                confidences.append(self.cxr_conf)
            else:
                raise ValueError(f"Unknown modality {m!r}")
        for per_tok, mask, conf_mod in zip(
            per_token_per_modality, token_masks, confidences,
        ):
            threshold = self.thresholds[self.modalities[len(high_pools)]]
            conf = conf_mod(per_tok)
            ph, pl, ch, cl = _confidence_partition(per_tok, conf, mask, threshold)
            # `_confidence_partition` returns (pooled_high, pooled_low,
            # conf_high, conf_low). Only the first two are feature vectors
            # (shape [B, D]) -- the last two are mean confidences (shape
            # [B]). Concat all four into the joint pool silently corrupts
            # the dimension (`[B, n_mods*(D+1)]` instead of `[B,
            # n_mods*D]`), which the high/low heads then reject. The
            # mean confidences are surfaced for logging/inspection but
            # not consumed by the fusion heads.
            high_pools.append(ph)
            low_pools.append(pl)
            high_confs.append(ch)
            low_confs.append(cl)

        high_pool = torch.cat(high_pools, dim=-1)
        low_pool = torch.cat(low_pools, dim=-1)

        present_mask = torch.stack(present_list, dim=1)  # [B, n_mods]

        fusion_out = self.fusion(
            pooled_per_modality=pooled_per_modality,
            high_pool=high_pool,
            low_pool=low_pool,
            present_mask=present_mask,
        )

        late_logit = fusion_out["late_logit"]  # [B]

        if "label" in batch:
            y = batch["label"].float().to(self.device_).view(-1)
            loss_aux, components = self._aux_losses(fusion_out, y, present_mask)
            self.last_loss_aux = loss_aux  # not detached; future train.py patch can .backward()
            self.last_aux_components = components
        else:
            self.last_loss_aux = torch.zeros((), device=self.device_)
            self.last_aux_components = {}

        return late_logit

    def _aux_losses(
        self, fusion_out: dict, y: torch.Tensor, present_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Auxiliary BCEs over the component logits. Weighted by the
        reference's `weight_high` / `weight_low` / `weight_late`
        (`arguments.py:109-111`). The primary late-fusion loss is left to
        train.py -- this is auxiliary supervision only.
        """
        bce = nn.BCEWithLogitsLoss()
        loss_late = bce(fusion_out["late_logit"], y)
        loss_high = bce(fusion_out["high_logit"], y)
        loss_low = bce(fusion_out["low_logit"], y)
        # Per-modality unimodal loss -- only over present samples per
        # modality, so a missing modality doesn't contribute noise.
        uni = fusion_out["uni"]  # [B, n_mods]
        uni_losses = []
        for i in range(uni.size(1)):
            mask_i = present_mask[:, i]
            if mask_i.any():
                # Reduce over only the present samples.
                l = F.binary_cross_entropy_with_logits(
                    uni[mask_i, i], y[mask_i], reduction="mean"
                )
                uni_losses.append(l)
        loss_uni = (
            torch.stack(uni_losses).mean()
            if uni_losses
            else torch.zeros((), device=uni.device)
        )
        loss_miss = bce(fusion_out["miss_logit"], y)

        total = (
            loss_late
            + self.weight_high * loss_high
            + self.weight_low * loss_low
            + loss_uni
            + loss_miss
        )
        components = {
            "loss_late": loss_late.detach(),
            "loss_high": loss_high.detach(),
            "loss_low": loss_low.detach(),
            "loss_uni": loss_uni.detach() if isinstance(loss_uni, torch.Tensor) else torch.tensor(loss_uni),
            "loss_miss": loss_miss.detach(),
        }
        return total, components


# ============================================================
# Standalone smoke test.
# ============================================================

if __name__ == "__main__":
    import tempfile
    from PIL import Image

    torch.manual_seed(0)
    B, T = 4, 48
    n_vars = 17

    print("--- 1. MedPatchFusion standalone ---")
    fusion = MedPatchFusion(n_modalities=2, hidden_size=32)
    p1, p2 = torch.randn(B, 32), torch.randn(B, 32)
    high = torch.randn(B, 64)
    low = torch.randn(B, 64)
    present = torch.tensor([[True, True]] * B)
    out = fusion([p1, p2], high, low, present)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert out["late_logit"].shape == (B,), "late_logit should be [B]"

    print("\n--- 2. MedPatchBaseline forward (mock batch, [ts, notes]) ---")
    mock_config = {
        "modalities": ["ts", "notes"],
        "model_args": {"hidden_size": 32, "lookback_hours": T},
    }
    adapter = MedPatchBaseline(mock_config, device="cpu")

    T_raw = 200
    admission_hour_now = 60.0
    mock_batch_notes_only = {
        "t_hours": torch.full((B,), admission_hour_now),
        "ts": {
            "value": torch.randn(B, T_raw),
            "var_idx": torch.randint(0, n_vars, (B, T_raw)),
            "mask": torch.ones(B, T_raw),
            "hours": torch.rand(B, T_raw) * admission_hour_now,
        },
        # SepsisDataset's collate_sepsis_batch gives `text` (list[list[str]])
        # and `path` (list[list[str]]); we don't have a real notes.parquet
        # here so we pass mock strings.
        "notes": {
            "text": [["patient stable on room air"], ["intubated for respiratory failure"]] * (B // 2)
                    + [["started on vancomycin", "cxr shows bilateral infiltrates"]] * (B // 2),
            "path": [[], [], [], []],
            "hours": torch.tensor([[55.0], [50.0], [58.0], [40.0]]),
            "mask": torch.tensor([[True], [True], [True], [True]]),
        },
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    out_ts_notes = adapter(mock_batch_notes_only)
    print(f"  logits: {tuple(out_ts_notes.shape)} (expect ({B},))")
    assert out_ts_notes.shape == (B,), "logits must be [B]"
    assert torch.isfinite(adapter.last_loss_aux), "last_loss_aux must be finite"

    # Mirrors run_epoch's exact usage in train.py
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(out_ts_notes, mock_batch_notes_only["label"])
    probs = torch.sigmoid(out_ts_notes)
    print(f"  train.py-style BCE loss: {loss.item():.4f}")
    print(f"  sigmoid probs (sample 0): {probs[0].item():.4f}")
    assert torch.isfinite(loss)

    print("\n--- 3. MedPatchBaseline forward (mock batch, [ts, notes, cxr], empty CXR paths) ---")
    mock_config_full = {
        "modalities": ["ts", "notes", "cxr"],
        "model_args": {"hidden_size": 32, "lookback_hours": T},
    }
    adapter_full = MedPatchBaseline(mock_config_full, device="cpu")
    mock_batch_full = dict(mock_batch_notes_only)
    mock_batch_full["cxr"] = {
        "hours": torch.tensor([[55.0], [50.0], [0.0], [58.0]]),
        "path": [[], [], [], []],  # no CXR available
        "mask": torch.tensor([[False], [False], [False], [False]]),
    }
    out_full = adapter_full(mock_batch_full)
    print(f"  logits: {tuple(out_full.shape)}")
    assert out_full.shape == (B,)
    assert torch.isfinite(adapter_full.last_loss_aux)

    print("\n--- 4. MedPatchBaseline with one real CXR path ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        cxr_path = f"{tmpdir}/mock_cxr.png"
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(cxr_path)
        mock_batch_one_cxr = dict(mock_batch_notes_only)
        mock_batch_one_cxr["cxr"] = {
            "hours": torch.tensor([[55.0], [50.0], [0.0], [58.0]]),
            "path": [[cxr_path], [], [], [cxr_path]],
            "mask": torch.tensor([[True], [False], [False], [True]]),
        }
        adapter_full2 = MedPatchBaseline(mock_config_full, device="cpu")
        out_one_cxr = adapter_full2(mock_batch_one_cxr)
        print(f"  logits: {tuple(out_one_cxr.shape)}")
        assert out_one_cxr.shape == (B,)
        assert torch.isfinite(adapter_full2.last_loss_aux)

    print("\n--- 5. Each ablation arm runs ---")
    for ab in MedPatchFusion.ABLATIONS:
        cfg = {"modalities": ["ts", "notes"], "model_args": {"hidden_size": 32, "lookback_hours": T, "ablation": ab}}
        m = MedPatchBaseline(cfg, device="cpu")
        o = m(mock_batch_notes_only)
        assert o.shape == (B,), f"ablation={ab!r} failed shape"
        assert torch.isfinite(m.last_loss_aux), f"ablation={ab!r} aux loss not finite"
        print(f"  ablation={ab!r}: OK")

    print("\nAll MedPatch smoke tests passed.")
