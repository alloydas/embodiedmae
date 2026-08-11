"""Quantify and visualize EmbodiedMAE-4M modality alignment.

The four modalities are RGB, depth, point cloud, and numeric spline parameters.
For alignment metrics each modality is encoded independently with all of its
tokens visible.  This prevents cross-attention to the other modalities from
leaking the answer into cross-modal retrieval.  It is, however, an
out-of-training-distribution probe because training always retained at least one
token from every modality.

Two representation stages are analyzed:

* ``pre_encoder``: mean-pooled modality/position-aware input tokens.
* ``post_encoder``: the same single-modality stream after the shared encoder,
  mean-pooled over its tokens by default.

Example:

    python analyze_modality_alignment_4m.py \
        --checkpoint outputs/run/checkpoints/checkpoint_epoch_800.pth \
        --split val \
        --output_dir outputs/run/latent_alignment_ep800 \
        --device cuda

Outputs include JSON/CSV metrics, compressed embeddings, pairwise heatmaps, and
a 2-D latent-space plot.  UMAP is optional; PCA and t-SNE use dependencies that
are already present in the training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from embodied_mae_4m import embodied_mae_4m_base, embodied_mae_4m_small
from sorghum_dataset_4m import SorghumDataset4M


MODALITIES = ("rgb", "depth", "pc", "params")
DISPLAY_NAMES = {
    "rgb": "RGB",
    "depth": "Depth",
    "pc": "Point cloud",
    "params": "Spline params",
}
DEFAULT_GROUP_REGEX = r"^(.*)_\d+$"
EPS = 1e-12
MAX_PAIRWISE_SAMPLES = 5000


def _to_builtin(value):
    """Convert NumPy values recursively so json.dump can serialize them."""
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def infer_group_id(sample_name: str, group_regex: str = DEFAULT_GROUP_REGEX) -> str:
    """Infer a plant/group id from a sample name.

    With the default regex, ``Sorghum_5_04`` becomes ``Sorghum_5``.  An empty
    regex disables grouping and treats every sample as a distinct group.
    """
    if not group_regex:
        return sample_name
    match = re.search(group_regex, sample_name)
    if match is None:
        return sample_name
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def stratified_indices(
    sample_names: Sequence[str],
    max_samples: int,
    group_regex: str = DEFAULT_GROUP_REGEX,
    one_per_group: bool = False,
    views_per_group: int = 2,
    seed: Optional[int] = None,
) -> List[int]:
    """Select a balanced, reproducible set of plant/view indices.

    If the whole split fits under ``max_samples`` it is retained.  Otherwise,
    ``views_per_group`` views are drawn from enough randomly ordered groups to
    reach the requested cap.  This keeps group-aware retrieval meaningful while
    avoiding a prefix containing views from only a few plants.
    """
    if views_per_group <= 0:
        raise ValueError("views_per_group must be positive")
    buckets: Dict[str, List[int]] = defaultdict(list)
    for index, name in enumerate(sample_names):
        buckets[infer_group_id(name, group_regex)].append(index)

    groups = sorted(buckets)
    if seed is not None:
        generator = np.random.default_rng(seed)
        generator.shuffle(groups)
        for group in groups:
            generator.shuffle(buckets[group])
    if one_per_group:
        selected_groups = groups
        group_view_limit = 1
    elif max_samples <= 0 or max_samples >= len(sample_names):
        selected_groups = groups
        group_view_limit = max(len(bucket) for bucket in buckets.values())
    else:
        selected_groups = []
        capacity = 0
        for group in groups:
            selected_groups.append(group)
            capacity += min(views_per_group, len(buckets[group]))
            if capacity >= max_samples:
                break
        group_view_limit = views_per_group

    selected = []
    for view_index in range(group_view_limit):
        for group in selected_groups:
            if view_index < len(buckets[group]):
                selected.append(buckets[group][view_index])

    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def _seed_worker(worker_id: int) -> None:
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def _masked_mean(
    tokens: torch.Tensor, valid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if valid is None:
        return tokens.mean(dim=1)
    weights = valid.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def extract_unimodal_latents(
    model,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    pc: torch.Tensor,
    params: torch.Tensor,
    text_valid: torch.Tensor,
    pool: str = "mean",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return independently encoded sample vectors for one paired batch.

    The model's native ``forward_encoder`` is deliberately not used: even with
    mask_ratio=0 it enforces masking, concatenates modalities, and allows
    cross-modal attention.  Here each stream receives the shared encoder alone.
    """
    if pool not in {"mean", "cls"}:
        raise ValueError(f"Unsupported pool={pool!r}; expected 'mean' or 'cls'")

    token_streams = {
        "rgb": (
            model.rgb_embed(rgb) + model.pos_embed_2d + model.modality_embed_rgb
        ),
        "depth": (
            model.depth_embed(depth)
            + model.pos_embed_2d
            + model.modality_embed_depth
        ),
        "pc": model.pc_embed(pc) + model.pos_embed_pc + model.modality_embed_pc,
        "params": (
            model.param_embed(params)
            + model.pos_embed_text
            + model.modality_embed_text
        ),
    }

    result: Dict[str, Dict[str, torch.Tensor]] = {
        "pre_encoder": {},
        "post_encoder": {},
    }
    batch_size = rgb.shape[0]

    for modality, tokens in token_streams.items():
        valid = text_valid if modality == "params" else None
        # Pre-encoder has no sample-dependent CLS representation, so token mean
        # is used regardless of the requested post-encoder pooling strategy.
        result["pre_encoder"][modality] = _masked_mean(tokens, valid)

        cls = model.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat([cls, tokens], dim=1)
        for block in model.encoder_blocks:
            encoded = block(encoded)
        encoded = model.encoder_norm(encoded)

        if pool == "cls":
            pooled = encoded[:, 0]
        else:
            pooled = _masked_mean(encoded[:, 1:], valid)
        result["post_encoder"][modality] = pooled

    return result


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def _center_and_normalize(x: np.ndarray) -> np.ndarray:
    return _l2_normalize(x - x.mean(axis=0, keepdims=True))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Biased linear centered-kernel alignment over paired samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be 2-D and have the same sample count")
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two samples")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    gram_x = x @ x.T
    gram_y = y @ y.T
    numerator = float(np.sum(gram_x * gram_y))
    denominator = math.sqrt(
        float(np.sum(gram_x * gram_x)) * float(np.sum(gram_y * gram_y))
    )
    if denominator == 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _random_recall_baseline(num_gallery: int, positives: Sequence[int], k: int) -> float:
    """Expected recall@k when ranking gallery items uniformly at random."""
    k = min(k, num_gallery)
    probabilities = []
    for n_positive in positives:
        if n_positive <= 0:
            probabilities.append(0.0)
            continue
        if k > num_gallery - n_positive:
            probabilities.append(1.0)
            continue
        no_hit = 1.0
        for offset in range(k):
            no_hit *= (num_gallery - n_positive - offset) / (num_gallery - offset)
        probabilities.append(1.0 - no_hit)
    return float(np.mean(probabilities))


def _tie_block_statistics(
    n_higher: int,
    n_tied: int,
    n_positive: int,
    recall_ks: Sequence[int],
) -> Dict[str, object]:
    """Expected rank metrics under uniform ordering within a score tie.

    The first relevant item lies after ``n_higher`` strictly better items.  Its
    position within the tied block is averaged over every possible tie ordering,
    removing dependence on gallery/sample order.
    """
    if not 0 < n_positive <= n_tied:
        raise ValueError("A tie block must contain at least one positive")

    expected_rank = n_higher + (n_tied + 1.0) / (n_positive + 1.0)
    expected_reciprocal_rank = 0.0
    survival = 1.0
    for offset in range(n_tied - n_positive + 1):
        denominator = n_tied - offset
        first_positive_probability = survival * n_positive / denominator
        expected_reciprocal_rank += first_positive_probability / (
            n_higher + offset + 1
        )
        survival *= (n_tied - n_positive - offset) / denominator

    recalls = {}
    for requested_k in recall_ks:
        slots = min(max(int(requested_k) - n_higher, 0), n_tied)
        if slots <= 0:
            probability = 0.0
        elif slots > n_tied - n_positive:
            probability = 1.0
        else:
            no_hit = 1.0
            for offset in range(slots):
                no_hit *= (n_tied - n_positive - offset) / (n_tied - offset)
            probability = 1.0 - no_hit
        recalls[int(requested_k)] = probability
    return {
        "expected_rank": expected_rank,
        "expected_reciprocal_rank": expected_reciprocal_rank,
        "recall": recalls,
    }


def retrieval_direction(
    similarities: np.ndarray,
    query_groups: Sequence[str],
    gallery_groups: Sequence[str],
    recall_ks: Sequence[int] = (1, 5),
) -> Dict[str, object]:
    """Compute exact-sample and group-aware retrieval in one direction."""
    similarities = np.asarray(similarities)
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("Retrieval expects a square paired similarity matrix")
    n_samples = similarities.shape[0]
    if len(query_groups) != n_samples or len(gallery_groups) != n_samples:
        raise ValueError("Group labels must match the similarity matrix size")

    exact_ranks = np.empty(n_samples, dtype=np.float64)
    exact_reciprocal_ranks = np.empty(n_samples, dtype=np.float64)
    group_ranks = np.empty(n_samples, dtype=np.float64)
    group_reciprocal_ranks = np.empty(n_samples, dtype=np.float64)
    exact_recalls = {int(k): np.empty(n_samples) for k in recall_ks}
    group_recalls = {int(k): np.empty(n_samples) for k in recall_ks}
    group_positive_counts = []

    gallery_groups_array = np.asarray(gallery_groups, dtype=str)
    for query_index in range(n_samples):
        row = similarities[query_index]
        target_score = row[query_index]
        exact_tie = np.isclose(row, target_score, rtol=1e-7, atol=1e-8)
        exact_higher = int(np.sum((row > target_score) & ~exact_tie))
        exact_stats = _tie_block_statistics(
            exact_higher, int(exact_tie.sum()), 1, recall_ks
        )
        exact_ranks[query_index] = exact_stats["expected_rank"]
        exact_reciprocal_ranks[query_index] = exact_stats[
            "expected_reciprocal_rank"
        ]
        for k in recall_ks:
            exact_recalls[int(k)][query_index] = exact_stats["recall"][int(k)]

        positives = gallery_groups_array == str(query_groups[query_index])
        n_group_positives = int(positives.sum())
        group_positive_counts.append(n_group_positives)
        if n_group_positives == 0:
            raise ValueError("Every query must have at least one positive gallery group")
        best_group_score = float(row[positives].max())
        group_tie = np.isclose(row, best_group_score, rtol=1e-7, atol=1e-8)
        group_higher = int(np.sum((row > best_group_score) & ~group_tie))
        group_stats = _tie_block_statistics(
            group_higher,
            int(group_tie.sum()),
            int(np.sum(group_tie & positives)),
            recall_ks,
        )
        group_ranks[query_index] = group_stats["expected_rank"]
        group_reciprocal_ranks[query_index] = group_stats[
            "expected_reciprocal_rank"
        ]
        for k in recall_ks:
            group_recalls[int(k)][query_index] = group_stats["recall"][int(k)]

    result = {
        "exact": {
            "median_rank": float(np.median(exact_ranks)),
            "mean_reciprocal_rank": float(exact_reciprocal_ranks.mean()),
        },
        "group": {
            "median_rank": float(np.median(group_ranks)),
            "mean_reciprocal_rank": float(group_reciprocal_ranks.mean()),
        },
    }
    for requested_k in recall_ks:
        k = min(int(requested_k), n_samples)
        key = f"recall_at_{requested_k}"
        result["exact"][key] = float(exact_recalls[int(requested_k)].mean())
        result["group"][key] = float(group_recalls[int(requested_k)].mean())
        result["exact"][f"random_{key}"] = _random_recall_baseline(
            n_samples, [1] * n_samples, k
        )
        result["group"][f"random_{key}"] = _random_recall_baseline(
            n_samples, group_positive_counts, k
        )
    return result


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def cosine_statistics(
    similarities: np.ndarray, groups: Sequence[str]
) -> Dict[str, float]:
    """Compare paired cosine with same-group and different-group negatives."""
    similarities = np.asarray(similarities)
    n_samples = similarities.shape[0]
    group_array = np.asarray(groups, dtype=str)
    diagonal = np.eye(n_samples, dtype=bool)
    same_group = group_array[:, None] == group_array[None, :]
    same_group_non_pair = same_group & ~diagonal
    different_group = ~same_group

    paired = np.diag(similarities)
    different_mean = _mean_or_nan(similarities[different_group])
    return {
        "paired_mean": float(paired.mean()),
        "paired_std": float(paired.std()),
        "same_group_non_pair_mean": _mean_or_nan(similarities[same_group_non_pair]),
        "different_group_mean": different_mean,
        "paired_minus_different_group": float(paired.mean() - different_mean),
    }


def pair_metrics(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    modality_a: str,
    modality_b: str,
    recall_ks: Sequence[int] = (1, 5),
) -> Dict[str, object]:
    if x.shape[0] != y.shape[0] or x.shape[0] != len(groups):
        raise ValueError("Pair metrics require aligned samples and group labels")

    raw_x, raw_y = _l2_normalize(x), _l2_normalize(y)
    centered_x, centered_y = _center_and_normalize(x), _center_and_normalize(y)
    spaces = {
        "raw": raw_x @ raw_y.T,
        "centered": centered_x @ centered_y.T,
    }
    result: Dict[str, object] = {"linear_cka": linear_cka(x, y)}

    for space_name, similarities in spaces.items():
        a_to_b = retrieval_direction(similarities, groups, groups, recall_ks)
        b_to_a = retrieval_direction(similarities.T, groups, groups, recall_ks)
        bidirectional = {}
        for target in ("exact", "group"):
            bidirectional[target] = {
                "median_rank": float(
                    np.mean(
                        [
                            a_to_b[target]["median_rank"],
                            b_to_a[target]["median_rank"],
                        ]
                    )
                ),
                "mean_reciprocal_rank": float(
                    np.mean(
                        [
                            a_to_b[target]["mean_reciprocal_rank"],
                            b_to_a[target]["mean_reciprocal_rank"],
                        ]
                    )
                ),
            }
            for k in recall_ks:
                key = f"recall_at_{k}"
                bidirectional[target][key] = float(
                    np.mean([a_to_b[target][key], b_to_a[target][key]])
                )
                bidirectional[target][f"random_{key}"] = float(
                    np.mean(
                        [
                            a_to_b[target][f"random_{key}"],
                            b_to_a[target][f"random_{key}"],
                        ]
                    )
                )

        result[space_name] = {
            "cosine": cosine_statistics(similarities, groups),
            "retrieval": {
                f"{modality_a}_to_{modality_b}": a_to_b,
                f"{modality_b}_to_{modality_a}": b_to_a,
                "bidirectional_mean": bidirectional,
            },
        }
    return result


def analyze_stage(
    embeddings: Mapping[str, np.ndarray],
    groups: Sequence[str],
    recall_ks: Sequence[int],
) -> Dict[str, object]:
    pairs = {}
    for index, modality_a in enumerate(MODALITIES):
        for modality_b in MODALITIES[index + 1 :]:
            key = f"{modality_a}__{modality_b}"
            pairs[key] = pair_metrics(
                embeddings[modality_a],
                embeddings[modality_b],
                groups,
                modality_a,
                modality_b,
                recall_ks,
            )
    return {"pairs": pairs}


def _summary_matrices(stage_metrics: Mapping[str, object], recall_k: int = 1):
    n_modalities = len(MODALITIES)
    cka = np.eye(n_modalities, dtype=np.float64)
    cosine_gap = np.full((n_modalities, n_modalities), np.nan, dtype=np.float64)
    exact_retrieval = np.eye(n_modalities, dtype=np.float64)
    group_retrieval = np.eye(n_modalities, dtype=np.float64)

    for i, modality_a in enumerate(MODALITIES):
        for j in range(i + 1, n_modalities):
            modality_b = MODALITIES[j]
            pair = stage_metrics["pairs"][f"{modality_a}__{modality_b}"]
            cka[i, j] = cka[j, i] = pair["linear_cka"]
            gap = pair["centered"]["cosine"]["paired_minus_different_group"]
            cosine_gap[i, j] = cosine_gap[j, i] = gap
            retrieval = pair["centered"]["retrieval"]["bidirectional_mean"]
            exact = retrieval["exact"][f"recall_at_{recall_k}"]
            group = retrieval["group"][f"recall_at_{recall_k}"]
            exact_retrieval[i, j] = exact_retrieval[j, i] = exact
            group_retrieval[i, j] = group_retrieval[j, i] = group

    return {
        "Linear CKA": cka,
        "Centered cosine gap\n(pair - different plant)": cosine_gap,
        f"Exact-view retrieval R@{recall_k}\n(centered, bidirectional)": exact_retrieval,
        f"Plant/group retrieval R@{recall_k}\n(centered, bidirectional)": group_retrieval,
    }


def _load_pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "embodiedmae-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pairwise_heatmaps(
    stage_metrics: Mapping[str, object],
    output_path: Path,
    stage_name: str,
    recall_k: int = 1,
) -> None:
    plt = _load_pyplot()
    matrices = _summary_matrices(stage_metrics, recall_k)
    labels = [DISPLAY_NAMES[name] for name in MODALITIES]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for ax, (title, matrix) in zip(axes.flat, matrices.items()):
        if "cosine gap" in title:
            finite = matrix[np.isfinite(matrix)]
            limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 0.05)
            image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text = "—" if not np.isfinite(value) else f"{value:.3f}"
                if not np.isfinite(value):
                    text_color = "black"
                elif "cosine gap" in title:
                    text_color = "black" if abs(value) < 0.45 * limit else "white"
                else:
                    text_color = "black" if value >= 0.65 else "white"
                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"EmbodiedMAE-4M modality alignment — {stage_name.replace('_', ' ')}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def project_2d(matrix: np.ndarray, method: str, seed: int) -> Tuple[np.ndarray, str]:
    """Project rows to 2-D, using optional dependencies only when requested."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if method == "pca":
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        if singular_values.size < 2:
            raise ValueError("PCA visualization requires at least two components")
        return u[:, :2] * singular_values[:2], "pca"
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, (matrix.shape[0] - 1) / 3.0))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=seed,
        ).fit_transform(matrix)
        return coords, "tsne"
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP was requested but umap-learn is not installed. "
                "Use --projection pca/tsne or install umap-learn."
            ) from exc
        n_neighbors = min(15, max(2, matrix.shape[0] - 1))
        coords = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        ).fit_transform(matrix)
        return coords, "umap"
    raise ValueError(f"Unsupported projection method: {method}")


def _plot_projection_panel(
    ax,
    coords_by_modality: Mapping[str, np.ndarray],
    groups: Sequence[str],
    sample_indices: np.ndarray,
    color_mode: str,
    title: str,
):
    plt = _load_pyplot()
    markers = {"rgb": "o", "depth": "s", "pc": "^", "params": "D"}
    modality_colors = {
        "rgb": "#d62728",
        "depth": "#1f77b4",
        "pc": "#2ca02c",
        "params": "#9467bd",
    }
    unique_groups = sorted(set(groups))
    group_index = {group: index for index, group in enumerate(unique_groups)}
    cmap = plt.get_cmap("tab20" if len(unique_groups) <= 20 else "turbo")
    denomin = max(len(unique_groups) - 1, 1)

    for sample_index in sample_indices:
        line = np.stack(
            [coords_by_modality[modality][sample_index] for modality in MODALITIES]
        )
        ax.plot(line[:, 0], line[:, 1], color="0.55", alpha=0.10, linewidth=0.6)

    for modality in MODALITIES:
        coords = coords_by_modality[modality][sample_indices]
        if color_mode == "modality":
            colors = modality_colors[modality]
            label = DISPLAY_NAMES[modality]
        else:
            colors = [
                cmap(group_index[str(groups[index])] / denomin) for index in sample_indices
            ]
            label = DISPLAY_NAMES[modality]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            marker=markers[modality],
            s=30,
            alpha=0.82,
            linewidths=0.25,
            edgecolors="white",
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)


def plot_latent_space(
    embeddings: Mapping[str, np.ndarray],
    groups: Sequence[str],
    output_path: Path,
    stage_name: str,
    method: str,
    seed: int,
    max_plot_samples: int,
) -> str:
    plt = _load_pyplot()
    n_samples = len(groups)
    if max_plot_samples > 0 and n_samples > max_plot_samples:
        sample_indices = np.linspace(
            0, n_samples - 1, num=max_plot_samples, dtype=np.int64
        )
        sample_indices = np.unique(sample_indices)
    else:
        sample_indices = np.arange(n_samples)

    plot_embeddings = {
        modality: embeddings[modality][sample_indices] for modality in MODALITIES
    }
    plot_groups = [groups[index] for index in sample_indices]
    n_plot_samples = len(sample_indices)
    plot_indices = np.arange(n_plot_samples)

    raw_stack = np.concatenate(
        [_l2_normalize(plot_embeddings[m]) for m in MODALITIES]
    )
    centered_stack = np.concatenate(
        [_center_and_normalize(plot_embeddings[m]) for m in MODALITIES]
    )
    raw_coords, actual_method = project_2d(raw_stack, method, seed)
    centered_coords, _ = project_2d(centered_stack, method, seed)

    def split(coords):
        return {
            modality: coords[
                index * n_plot_samples : (index + 1) * n_plot_samples
            ]
            for index, modality in enumerate(MODALITIES)
        }

    raw_by_modality = split(raw_coords)
    centered_by_modality = split(centered_coords)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    _plot_projection_panel(
        axes[0, 0],
        raw_by_modality,
        plot_groups,
        plot_indices,
        "modality",
        "L2-normalized representations — color = modality",
    )
    _plot_projection_panel(
        axes[0, 1],
        raw_by_modality,
        plot_groups,
        plot_indices,
        "group",
        "L2-normalized — color = plant, marker = modality",
    )
    _plot_projection_panel(
        axes[1, 0],
        centered_by_modality,
        plot_groups,
        plot_indices,
        "modality",
        "Per-modality centered + L2-normalized — color = modality",
    )
    _plot_projection_panel(
        axes[1, 1],
        centered_by_modality,
        plot_groups,
        plot_indices,
        "group",
        "Centered + L2-normalized — color = plant, marker = modality",
    )
    fig.suptitle(
        f"{stage_name.replace('_', ' ').title()} latent space ({actual_method.upper()})\n"
        "Gray lines connect the same paired sample across modalities",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return actual_method


def metrics_rows(
    all_metrics: Mapping[str, Mapping[str, object]], recall_ks: Sequence[int]
) -> List[Dict[str, object]]:
    rows = []
    for stage_name, stage_metrics in all_metrics.items():
        for pair_name, pair in stage_metrics["pairs"].items():
            modality_a, modality_b = pair_name.split("__")
            row = {
                "stage": stage_name,
                "modality_a": modality_a,
                "modality_b": modality_b,
                "linear_cka": pair["linear_cka"],
            }
            for space_name in ("raw", "centered"):
                cosine = pair[space_name]["cosine"]
                for key, value in cosine.items():
                    row[f"{space_name}_cosine_{key}"] = value
                bidirectional = pair[space_name]["retrieval"]["bidirectional_mean"]
                for target in ("exact", "group"):
                    row[f"{space_name}_{target}_median_rank"] = bidirectional[target][
                        "median_rank"
                    ]
                    row[f"{space_name}_{target}_mrr"] = bidirectional[target][
                        "mean_reciprocal_rank"
                    ]
                    for k in recall_ks:
                        key = f"recall_at_{k}"
                        row[f"{space_name}_{target}_r@{k}"] = bidirectional[target][key]
                        row[f"{space_name}_{target}_random_r@{k}"] = bidirectional[
                            target
                        ][f"random_{key}"]
            rows.append(row)
    return rows


def save_metrics_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def saved_run_config_path(checkpoint_path: Path) -> Optional[Path]:
    """Return the immutable config saved beside a checkpoint, when available."""
    run_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    for filename in ("config.json", "config_used.json", "config.yaml", "config.yml"):
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_config_path(
    checkpoint_path: Path, explicit_config: Optional[Path]
) -> Tuple[Path, bool]:
    """Choose explicit config, then the run-saved config, then repo fallback."""
    if explicit_config is not None:
        return explicit_config, False
    saved_config = saved_run_config_path(checkpoint_path)
    if saved_config is not None:
        return saved_config, True
    fallback = Path("config_4m.yaml")
    if not fallback.is_file():
        raise FileNotFoundError(
            "No --config was provided, no config was saved beside the checkpoint, "
            "and config_4m.yaml does not exist."
        )
    return fallback, False


def normalize_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Accept both nested YAML configs and the flat config.json saved by training."""
    if isinstance(config.get("data"), Mapping) and isinstance(
        config.get("model"), Mapping
    ):
        return dict(config)

    required = ("data_root", "img_size", "num_points", "model_size")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
    data_keys = ("data_root", "img_size", "num_points")
    model_keys = (
        "model_size",
        "patch_size",
        "num_pc_tokens",
        "pc_group_size",
        "mask_ratio",
        "pc_loss_weight",
        "loss_name",
        "qal_threshold",
        "qal_alpha",
        "qal_use_squared",
        "max_leaves",
        "spline_loss_weight",
        "depth_norm_type",
        "pc_deterministic_fps",
        "pc_add_center_coordinates",
    )
    return {
        "data": {key: config[key] for key in data_keys if key in config},
        "model": {key: config[key] for key in model_keys if key in config},
    }


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open() as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return normalize_config(raw_config)


def build_model_from_config(config: Mapping[str, object]):
    data_config = config["data"]
    model_config = config["model"]
    model_size = model_config.get("model_size", "base")
    build_fn = embodied_mae_4m_small if model_size == "small" else embodied_mae_4m_base
    return build_fn(
        img_size=data_config.get("img_size", 224),
        patch_size=model_config.get("patch_size", 16),
        num_pc_tokens=model_config.get("num_pc_tokens", 196),
        pc_group_size=model_config.get("pc_group_size", 32),
        target_points=data_config.get("num_points", 8196),
        pc_loss_weight=model_config.get("pc_loss_weight", 10.0),
        pc_loss_name=model_config.get("loss_name", "chamfer"),
        qal_threshold=model_config.get("qal_threshold", 0.01),
        qal_alpha=model_config.get("qal_alpha", 100.0),
        qal_use_squared=model_config.get("qal_use_squared", False),
        # No decoder loss is evaluated.  Keeping this at zero avoids importing
        # and initializing GeomLoss without changing checkpoint parameters.
        sinkhorn_loss_weight=0.0,
        max_leaves=model_config.get("max_leaves", 24),
        spline_loss_weight=model_config.get("spline_loss_weight", 5.0),
        depth_norm_type=model_config.get("depth_norm_type", "minmax"),
        pc_deterministic_fps=model_config.get("pc_deterministic_fps", False),
        pc_add_center_coordinates=model_config.get(
            "pc_add_center_coordinates", False),
    )


def load_checkpoint_model(
    checkpoint_path: Path, config: Mapping[str, object], device: torch.device
):
    model = build_model_from_config(config)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_info = {
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        }
    else:
        state_dict = checkpoint
        checkpoint_info = {"epoch": None, "best_val_loss": None}
    if next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    del checkpoint, state_dict
    model.to(device)
    model.eval()
    return model, checkpoint_info


def collect_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
    pool: str,
    use_amp: bool,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    chunks = {
        stage: {modality: [] for modality in MODALITIES}
        for stage in ("pre_encoder", "post_encoder")
    }
    sample_names: List[str] = []
    amp_enabled = use_amp and device.type == "cuda"

    for rgb, depth, pc, params, text_valid, names in tqdm(
        loader, desc="Extracting unimodal latents"
    ):
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        pc = pc.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        text_valid = text_valid.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            batch = extract_unimodal_latents(
                model, rgb, depth, pc, params, text_valid, pool=pool
            )
        for stage in chunks:
            for modality in MODALITIES:
                chunks[stage][modality].append(
                    batch[stage][modality].detach().float().cpu().numpy()
                )
        sample_names.extend(str(name) for name in names)

    embeddings = {
        stage: {
            modality: np.concatenate(modality_chunks, axis=0)
            for modality, modality_chunks in stage_chunks.items()
        }
        for stage, stage_chunks in chunks.items()
    }
    return embeddings, sample_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CKA, cosine alignment, retrieval, and latent-space "
        "structure for EmbodiedMAE-4M."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config override; defaults to config.json saved beside the checkpoint.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Maximum stratified samples (default: 1000); 0 uses splits up to 5000.",
    )
    parser.add_argument(
        "--one_per_group",
        action="store_true",
        help="Use one view per inferred plant/group (recommended on train).",
    )
    parser.add_argument(
        "--views_per_group",
        type=int,
        default=2,
        help="Views sampled per plant unless the whole split fits (default: 2).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", choices=("mean", "cls"), default="mean")
    parser.add_argument(
        "--group_regex",
        default=DEFAULT_GROUP_REGEX,
        help="Regex whose first capture group is the plant id; empty disables grouping.",
    )
    parser.add_argument("--recall_k", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--projection", choices=("pca", "tsne", "umap", "none"), default="pca"
    )
    parser.add_argument("--max_plot_samples", type=int, default=100)
    parser.add_argument(
        "--amp", action="store_true", help="Use float16 autocast during CUDA extraction."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 0:
        raise ValueError("--num_samples must be zero or positive")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch_size must be positive and --num_workers non-negative")
    if args.max_plot_samples < 0:
        raise ValueError("--max_plot_samples must be zero or positive")
    if args.views_per_group <= 0:
        raise ValueError("--views_per_group must be positive")
    recall_ks = tuple(sorted(set(int(value) for value in args.recall_k)))
    if not recall_ks or any(value <= 0 for value in recall_ks):
        raise ValueError("--recall_k values must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path, using_saved_config = resolve_config_path(args.checkpoint, args.config)
    config = load_config(config_path)
    data_root = args.data_root or Path(config["data"]["data_root"])

    training_config_path = saved_run_config_path(args.checkpoint)
    training_data_root = None
    if training_config_path is not None:
        training_config = load_config(training_config_path)
        training_data_root = Path(training_config["data"]["data_root"])
    data_root_mismatch = (
        training_data_root is not None
        and Path(data_root).resolve() != training_data_root.resolve()
    )
    if data_root_mismatch:
        print(
            "Warning: evaluation data_root differs from the data_root saved with "
            f"the checkpoint:\n  train={training_data_root}\n  eval ={data_root}"
        )

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU (base-model extraction is slow).")
        device = torch.device("cpu")
    else:
        device = requested_device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = SorghumDataset4M(
        data_root,
        img_size=config["data"].get("img_size", 224),
        num_points=config["data"].get("num_points", 8196),
        split=args.split,
        max_leaves=config["model"].get("max_leaves", 24),
    )
    all_names = [folder.name for folder in dataset.samples]
    selected_indices = stratified_indices(
        all_names,
        args.num_samples,
        args.group_regex,
        one_per_group=args.one_per_group,
        views_per_group=args.views_per_group,
        seed=args.seed,
    )
    if len(selected_indices) < 2:
        raise ValueError("At least two samples are required for alignment analysis")
    if len(selected_indices) > MAX_PAIRWISE_SAMPLES:
        raise ValueError(
            "Pairwise retrieval and CKA scale quadratically. Selected "
            f"{len(selected_indices)} samples, but the safe limit is "
            f"{MAX_PAIRWISE_SAMPLES}; pass --num_samples {MAX_PAIRWISE_SAMPLES} "
            "or less."
        )
    selected = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    print(f"Using config: {config_path}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint_info = load_checkpoint_model(args.checkpoint, config, device)
    print(
        f"Selected {len(selected_indices)} samples from {len(dataset)} "
        f"({args.split}, stratified across plants); device={device}, pool={args.pool}"
    )
    embeddings, sample_names = collect_embeddings(
        model, loader, device, args.pool, args.amp
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    group_ids = [infer_group_id(name, args.group_regex) for name in sample_names]
    group_counts = Counter(group_ids)
    n_groups = len(group_counts)
    group_count_histogram = Counter(group_counts.values())
    if n_groups < 20:
        print(
            f"Warning: only {n_groups} independent groups for {len(sample_names)} samples; "
            "interpret CKA cautiously. Consider --split train --one_per_group."
        )

    metrics = {
        stage: analyze_stage(stage_embeddings, group_ids, recall_ks)
        for stage, stage_embeddings in embeddings.items()
    }
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_info.get("epoch"),
        "checkpoint_best_val_loss": checkpoint_info.get("best_val_loss"),
        "config": str(config_path),
        "using_checkpoint_saved_config": using_saved_config,
        "data_root": str(data_root),
        "checkpoint_training_data_root": (
            str(training_data_root) if training_data_root is not None else None
        ),
        "data_root_mismatch": data_root_mismatch,
        "split": args.split,
        "n_samples": len(sample_names),
        "n_groups": n_groups,
        "group_regex": args.group_regex,
        "one_per_group": args.one_per_group,
        "requested_views_per_group": (
            1 if args.one_per_group else args.views_per_group
        ),
        "actual_views_per_group": {
            "min": min(group_counts.values()),
            "max": max(group_counts.values()),
            "histogram": {
                str(views): count
                for views, count in sorted(group_count_histogram.items())
            },
        },
        "pool": args.pool,
        "seed": args.seed,
        "recall_k": list(recall_ks),
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "probe": "single-modality, all tokens visible, shared encoder",
        "probe_warning": (
            "This prevents cross-modal leakage but is outside the exact training mask "
            "distribution, which retained at least one visible token per modality."
        ),
        "params_padding_policy": (
            "Spline padding tokens remain in encoder attention to match training, but "
            "only text_valid tokens contribute to pre/post sample mean pooling."
        ),
        "interpretation": (
            "Use group-aware retrieval as primary when multiple views share identical "
            "spline parameters; exact-view retrieval is also reported for completeness."
        ),
        "retrieval_tie_policy": (
            "Metrics are expectations under uniform random ordering within equal-score "
            "blocks (rtol=1e-7, atol=1e-8)."
        ),
        "sample_names": sample_names,
        "group_ids": group_ids,
    }

    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(
            _to_builtin({"metadata": metadata, "representations": metrics}),
            handle,
            indent=2,
            allow_nan=False,
        )
    save_metrics_csv(metrics_rows(metrics, recall_ks), args.output_dir / "metrics.csv")

    arrays = {
        "sample_names": np.asarray(sample_names, dtype=str),
        "group_ids": np.asarray(group_ids, dtype=str),
    }
    for stage, stage_embeddings in embeddings.items():
        for modality, values in stage_embeddings.items():
            arrays[f"{stage}__{modality}"] = values.astype(np.float32)
    np.savez_compressed(args.output_dir / "embeddings.npz", **arrays)

    first_recall_k = recall_ks[0]
    for stage, stage_metrics in metrics.items():
        plot_pairwise_heatmaps(
            stage_metrics,
            args.output_dir / f"pairwise_metrics_{stage}.png",
            stage,
            recall_k=first_recall_k,
        )
        if args.projection != "none":
            plot_latent_space(
                embeddings[stage],
                group_ids,
                args.output_dir / f"latent_space_{stage}_{args.projection}.png",
                stage,
                args.projection,
                args.seed,
                args.max_plot_samples,
            )

    print(f"Saved alignment report to: {args.output_dir}")
    print("Primary readout: post_encoder plant/group retrieval and Linear CKA.")


if __name__ == "__main__":
    main()
