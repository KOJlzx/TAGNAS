"""Embedding generators used by the async runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Callable, Mapping, Sequence

import torch

from .types import CacheKey, EmbeddingRecord


class BaseGraphEmbeddingAdapter(ABC):
    """Inference-only graph embedding generator."""

    @abstractmethod
    def generate(
        self,
        node_ids: Sequence[int],
        subgraphs: Mapping[int, Sequence[int]],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        raise NotImplementedError()


class BaseTextEmbeddingAdapter(ABC):
    """Inference-only text embedding generator."""

    @abstractmethod
    def generate(
        self,
        node_ids: Sequence[int],
        texts: Mapping[int, str],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        raise NotImplementedError()


class CallableGraphEmbeddingAdapter(BaseGraphEmbeddingAdapter):
    """Wrapper reserved for integrating real GraphNAS/GNN modules later."""

    def __init__(self, fn: Callable[[Sequence[int], Mapping[int, Sequence[int]]], list[torch.Tensor]]) -> None:
        self.fn = fn

    def generate(
        self,
        node_ids: Sequence[int],
        subgraphs: Mapping[int, Sequence[int]],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        with torch.no_grad():
            tensors = self.fn(node_ids, subgraphs)

        records: list[EmbeddingRecord] = []
        for index, (node_id, tensor) in enumerate(zip(node_ids, tensors)):
            records.append(
                EmbeddingRecord(
                    key=CacheKey("graph", int(node_id)),
                    tensor=tensor.detach().clone(),
                    version=version_offset + index + 1,
                    producer=producer,
                )
            )
        return records


class CallableTextEmbeddingAdapter(BaseTextEmbeddingAdapter):
    """Wrapper reserved for integrating a real LLM sentence embedder later."""

    def __init__(self, fn: Callable[[Sequence[int], Mapping[int, str]], list[torch.Tensor]]) -> None:
        self.fn = fn

    def generate(
        self,
        node_ids: Sequence[int],
        texts: Mapping[int, str],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        with torch.no_grad():
            tensors = self.fn(node_ids, texts)

        records: list[EmbeddingRecord] = []
        for index, (node_id, tensor) in enumerate(zip(node_ids, tensors)):
            records.append(
                EmbeddingRecord(
                    key=CacheKey("text", int(node_id)),
                    tensor=tensor.detach().clone(),
                    version=version_offset + index + 1,
                    producer=producer,
                )
            )
        return records


class MockGraphEmbeddingAdapter(BaseGraphEmbeddingAdapter):
    """Deterministic graph embedder used by the runnable skeleton."""

    def __init__(self, embedding_dim: int, device: str = "cpu") -> None:
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

    def generate(
        self,
        node_ids: Sequence[int],
        subgraphs: Mapping[int, Sequence[int]],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        base = torch.linspace(0.0, 1.0, self.embedding_dim, device=self.device)
        with torch.no_grad():
            for index, node_id in enumerate(node_ids):
                neighbors = subgraphs.get(int(node_id), [int(node_id)])
                hop_size = max(len(neighbors), 1)
                tensor = torch.sin(base * (int(node_id) + 1.0))
                tensor = tensor + torch.cos(base * hop_size)
                tensor = tensor + hop_size * 0.01
                records.append(
                    EmbeddingRecord(
                        key=CacheKey("graph", int(node_id)),
                        tensor=tensor.detach().clone(),
                        version=version_offset + index + 1,
                        producer=producer,
                    )
                )
        return records


class MockTextEmbeddingAdapter(BaseTextEmbeddingAdapter):
    """Deterministic text embedder that can run on GPU or CPU."""

    def __init__(self, embedding_dim: int, device: str = "cpu") -> None:
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

    def generate(
        self,
        node_ids: Sequence[int],
        texts: Mapping[int, str],
        *,
        version_offset: int = 0,
        producer: str,
    ) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        base = torch.linspace(0.0, 1.0, self.embedding_dim, device=self.device)
        with torch.no_grad():
            for index, node_id in enumerate(node_ids):
                text = texts[int(node_id)]
                text_hash = self._stable_hash(text)
                scale = 1.0 + (text_hash % 13) * 0.01
                tensor = torch.cos(base * scale * (int(node_id) + 1.0))
                tensor = tensor + torch.sin(base * ((text_hash % 29) + 1))
                records.append(
                    EmbeddingRecord(
                        key=CacheKey("text", int(node_id)),
                        tensor=tensor.detach().clone(),
                        version=version_offset + index + 1,
                        producer=producer,
                    )
                )
        return records

    @staticmethod
    def _stable_hash(text: str) -> int:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
