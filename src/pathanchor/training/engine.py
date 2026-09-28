"""Training engine.

Ref: Sec. IV-G (the reported training budget and its four-accelerator profile),
Algorithm 1 (the score the engine optimises is the path marginal).

Every patient contributes the retained paths of one option: the regimen actually
administered when that option is feasible, and otherwise the guideline's priority
option. That choice is a declared engineering default -- the manuscript reports the
response target but not the training objective -- and it is what makes the
supervised term well defined, because the observed regimen's outcome is the one the
record actually carries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from pathanchor.evaluation.pipeline import EvidenceStore, PatientEvidence
from pathanchor.graph.index import CompactGraph
from pathanchor.losses.composite import CompositeObjective, LossBreakdown, LossWeights
from pathanchor.models.estimator import PathAnchorEstimator
from pathanchor.paths.scorer import PathBatch, encode_paths
from pathanchor.training.amp import PrecisionConfig, autocast_context, make_grad_scaler
from pathanchor.training.distributed import (
    DistributedContext,
    all_reduce_mean,
    shard_indices,
    unwrap,
)
from pathanchor.training.optim import OptimiserConfig, build_optimiser, clip_gradients
from pathanchor.training.schedule import ScheduleConfig, build_schedule
from pathanchor.training.seed import set_seed

EMPTY_PATH_FLOOR: float = -60.0


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 8
    grad_accumulation: int = 1
    max_steps: int = 2000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 100
    schedule: str = "cosine"
    grad_clip: float = 1.0
    seed: int = 20260101
    precision: str = "fp32"
    path_cap: int = 12
    epsilon: float = 1.0e-3
    ranking_scale: float = 0.1

    def optimiser(self) -> OptimiserConfig:
        return OptimiserConfig(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def scheduler(self) -> ScheduleConfig:
        return ScheduleConfig(
            name=self.schedule,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
        )

    def precision_config(self) -> PrecisionConfig:
        return PrecisionConfig(enabled=self.precision != "fp32", dtype=self.precision)


@dataclass(slots=True)
class TrainingHistory:
    losses: list[float] = field(default_factory=list)
    terms: dict[str, list[float]] = field(default_factory=dict)
    learning_rates: list[float] = field(default_factory=list)
    steps: int = 0
    epochs: int = 0
    seed: int = 0

    def note(self, breakdown: LossBreakdown, rate: float) -> None:
        self.losses.append(breakdown.total)
        for name, value in breakdown.terms.items():
            self.terms.setdefault(name, []).append(value)
        self.learning_rates.append(rate)

    def moved(self, window: int = 5) -> bool:
        if len(self.losses) < window * 2:
            return False
        head = float(np.mean(self.losses[:window]))
        tail = float(np.mean(self.losses[-window:]))
        return tail < head

    def summary(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "seed": self.seed,
            "first_loss": self.losses[0] if self.losses else 0.0,
            "final_loss": self.losses[-1] if self.losses else 0.0,
            "moved": self.moved(),
            "terms": {name: values[-1] for name, values in sorted(self.terms.items())},
        }


@dataclass(slots=True)
class EvidenceBatch:
    features: Tensor
    modality: Tensor
    labels: Tensor
    mask_braf: Tensor
    mask_wild: Tensor
    path_batch: PathBatch
    path_weights: Tensor
    path_mask: Tensor
    path_owner: Tensor
    size: int


def select_training_option(item: PatientEvidence, administered: str) -> int:
    """Index of the option the supervised term is attached to."""
    if not item.options:
        return -1
    if administered in item.options:
        return item.options.index(administered)
    return 0


def cap_paths(item: PatientEvidence, segment: int, cap: int) -> list[int]:
    """The highest-weight retained paths of one option, at most ``cap`` of them.

    The cap bounds a batch's tensor width; the selection is by evidence weight, so
    a route is dropped only when a stronger route to the same option replaces it.
    """
    positions = np.nonzero(item.segments == segment)[0]
    if positions.shape[0] <= cap:
        return [int(value) for value in positions.tolist()]
    order = sorted(
        (int(value) for value in positions.tolist()),
        key=lambda slot: (-item.option_weights[slot], slot),
    )
    return order[:cap]


def stack_path_batches(batches: Sequence[PathBatch]) -> PathBatch:
    return PathBatch(
        entity_indices=torch.cat([item.entity_indices for item in batches], dim=0),
        entity_types=torch.cat([item.entity_types for item in batches], dim=0),
        relation_ids=torch.cat([item.relation_ids for item in batches], dim=0),
        tier_ids=torch.cat([item.tier_ids for item in batches], dim=0),
        edge_weights=torch.cat([item.edge_weights for item in batches], dim=0),
        mask=torch.cat([item.mask for item in batches], dim=0),
        lengths=torch.cat([item.lengths for item in batches], dim=0),
    )


class TrainingEngine:
    """Fits the estimator on a stored split and reports each objective term."""

    def __init__(
        self,
        model: nn.Module,
        store: EvidenceStore,
        graph: CompactGraph,
        config: TrainConfig,
        weights: LossWeights | None = None,
        context: DistributedContext | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.graph = graph
        self.config = config
        self.context = context if context is not None else DistributedContext()
        self.objective = CompositeObjective(
            weights=weights if weights is not None else LossWeights(),
            epsilon=config.epsilon,
            ranking_scale=config.ranking_scale,
        )
        self.optimiser = build_optimiser(model, config.optimiser())
        self.schedule = build_schedule(self.optimiser, config.scheduler())
        self.device = self.context.device()
        self.scaler = make_grad_scaler(config.precision_config(), self.device)
        self.history = TrainingHistory(seed=config.seed)
        self._width = store.budget.max_hops

    def build_batch(self, positions: Sequence[int]) -> EvidenceBatch:
        """Assemble one batch, one padded path row per patient."""
        features: list[Tensor] = []
        modality: list[Tensor] = []
        labels: list[Tensor] = []
        mask_braf: list[Tensor] = []
        mask_wild: list[Tensor] = []
        weights: list[Tensor] = []
        masks: list[Tensor] = []
        per_patient: list[PathBatch] = []
        owners: list[int] = []
        width = self._width
        for position in positions:
            item = self.store.items[position]
            features.append(torch.as_tensor(item.features, dtype=torch.float32))
            modality.append(torch.as_tensor(item.modality, dtype=torch.float32))
            labels.append(torch.tensor(float(item.label), dtype=torch.float32))
            mask_braf.append(torch.tensor(1.0 if item.is_braf_v600 else 0.0))
            mask_wild.append(torch.tensor(0.0 if item.is_braf_v600 else 1.0))
            administered = self.store.records[position].treatment.value
            segment = select_training_option(item, administered)
            effective_cap = min(self.config.path_cap, width)
            chosen = cap_paths(item, segment, effective_cap) if segment >= 0 else []
            row_weight = np.zeros(width, dtype=np.float32)
            row_mask = np.zeros(width, dtype=np.float32)
            if chosen:
                for slot, index in enumerate(chosen):
                    row_weight[slot] = float(item.option_weights[index])
                    row_mask[slot] = 1.0
                subset = [item.paths[index] for index in chosen]
                per_patient.append(encode_paths(self.graph, subset, width, self.store.key_index))
                owners.extend([len(weights)] * len(chosen))
            else:
                per_patient.append(_empty_path_batch(width))
                owners.append(len(weights))
            weights.append(torch.as_tensor(row_weight))
            masks.append(torch.as_tensor(row_mask))
        return EvidenceBatch(
            features=torch.stack(features),
            modality=torch.stack(modality),
            labels=torch.stack(labels),
            mask_braf=torch.stack(mask_braf),
            mask_wild=torch.stack(mask_wild),
            path_batch=stack_path_batches(per_patient),
            path_weights=torch.stack(weights),
            path_mask=torch.stack(masks),
            path_owner=torch.as_tensor(owners, dtype=torch.long),
            size=len(positions),
        )

    def fit(self) -> TrainingHistory:
        set_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        order = np.asarray(shard_indices(self.store.size, self.context), dtype=np.int64)
        for epoch in range(self.config.epochs):
            if self.history.steps >= self.config.max_steps:
                break
            shuffled = np.array(order, copy=True)
            rng.shuffle(shuffled)
            epoch_losses: list[float] = []
            for start in range(0, shuffled.shape[0], self.config.batch_size):
                if self.history.steps >= self.config.max_steps:
                    break
                positions = [
                    int(value) for value in shuffled[start : start + self.config.batch_size]
                ]
                if not positions:
                    continue
                loss, breakdown = self.step(positions)
                epoch_losses.append(float(loss.detach()))
                self.optimiser_step(loss)
                self.history.note(breakdown, float(self.optimiser.param_groups[0]["lr"]))
                self.history.steps += 1
            self.history.epochs = epoch + 1
            self._record_epoch(epoch_losses)
        return self.history

    def _record_epoch(self, epoch_losses: Sequence[float]) -> None:
        if not epoch_losses:
            return
        mean_loss = all_reduce_mean(float(np.mean(epoch_losses)), self.context)
        self.history.terms.setdefault("epoch_mean", []).append(mean_loss)

    def step(self, positions: Sequence[int]) -> tuple[Tensor, LossBreakdown]:
        """Forward and objective evaluation, without the optimiser update."""
        self.model.train()
        self.optimiser.zero_grad(set_to_none=True)
        batch = self.build_batch(positions)
        core = cast(PathAnchorEstimator, unwrap(self.model))
        with autocast_context(self.config.precision_config(), self.device):
            encoded = core.encode(batch.features, batch.modality)
            expanded = encoded[batch.path_owner]
            flat = core.path_scores(batch.path_batch, expanded)
            rows = fold_rows(flat, batch.path_owner, self._width, batch.size)
            marginal = torch.logsumexp(rows, dim=-1)
            patient_scores = core.combine(core.base_logit(encoded), marginal)
            loss, breakdown = self.objective(
                patient_scores,
                batch.labels,
                batch.mask_braf,
                batch.mask_wild,
                rows,
                batch.path_weights,
                batch.path_mask,
            )
            loss = loss / float(max(self.config.grad_accumulation, 1))
        return (loss, breakdown)

    def optimiser_step(self, loss: Tensor) -> None:
        """Backward, clip and step. Separated so a check can assert the update."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            clip_gradients(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            torch.autograd.backward(loss)
            clip_gradients(self.model.parameters(), self.config.grad_clip)
            self.optimiser.step()
        self.schedule.step()


def _empty_path_batch(width: int) -> PathBatch:
    long_zeros = torch.zeros((1, width + 1), dtype=torch.long)
    short_zeros = torch.zeros((1, width), dtype=torch.long)
    return PathBatch(
        long_zeros,
        long_zeros,
        short_zeros,
        short_zeros,
        short_zeros.float(),
        short_zeros.float(),
        torch.zeros(1, dtype=torch.long),
    )


def fold_rows(
    flat: Tensor,
    patient_index: Tensor,
    width: int,
    batch_size: int,
) -> Tensor:
    """Fold the flat per-path score vector into one padded row per patient.

    The flat order is the batch order, so the fold is a sequential write per
    patient and lands in the same column order the weights and mask used.
    """
    rows = torch.full((batch_size, width), EMPTY_PATH_FLOOR, dtype=flat.dtype)
    cursor = [0] * batch_size
    for index in range(flat.shape[0]):
        row = int(patient_index[index])
        slot = cursor[row]
        if slot < width:
            rows[row, slot] = flat[index]
        cursor[row] += 1
    return rows


def summarise_history(history: TrainingHistory) -> Mapping[str, object]:
    return history.summary()
