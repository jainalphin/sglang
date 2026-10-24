from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import torch

from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

# State components the digest covers, grouped by how the transfer addresses
# their rows (see `digest`). The rest -- DSA_TAIL, QSA_PENDING, SWA_RING,
# DSV4_REQUEST_STATE, ... -- are addressed by descriptors only the transfer
# backend decodes, so both sides leave them out.
_PAGE_STATE_TYPES = frozenset(
    {
        StateType.BLOCK_SCALE,
        StateType.QSA_COMPRESSED,
        StateType.MINIMAX_INDEX_K,
        StateType.MINIMAX_DENSE_KV,
    }
)
_SWA_STATE_TYPES = frozenset({StateType.SWA, StateType.BLOCK_SCALE_SWA})
_DIGESTED_STATE_TYPES = (
    _PAGE_STATE_TYPES | _SWA_STATE_TYPES | {StateType.MAMBA, StateType.DSA}
)


def is_health_check_req(req: Req) -> bool:
    rid = req.rid
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def _page_indices(kv_indices: torch.Tensor, page_size: int) -> torch.Tensor:
    idx = kv_indices.to(torch.int64).reshape(-1)
    return (idx[::page_size] // page_size).contiguous()


class KvChecksumComputer:
    """Adler-32 over the KV a request hands from prefill to decode.

    Prefill digests before it sends the last chunk and decode digests when the
    request enters a batch. Both go through `digest`, so they read the same
    rows: the KV pages of tokens `[decode_prefix_len, seq_len)`, which is what
    the prefill transfers, plus the state rows addressed as the transfer
    addresses them.
    """

    def __init__(
        self,
        kv_data_ptrs: Sequence[int],
        kv_item_lens: Sequence[int],
        page_size: int = 1,
        state_types: Sequence[StateType] = (),
        state_data_ptrs: Sequence[Sequence[int]] = (),
        state_item_lens: Sequence[Sequence[int]] = (),
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        self.page_size = page_size
        self._state = [
            (state_type, [int(p) for p in ptrs], [int(n) for n in lens])
            for state_type, ptrs, lens in zip(
                state_types, state_data_ptrs, state_item_lens
            )
            if state_type in _DIGESTED_STATE_TYPES
        ]
        for _, ptrs, lens in self._state:
            assert len(ptrs) == len(lens)
        self.signature = self._layout_signature()

    @classmethod
    def from_kv_args(cls, kv_args: KVArgs) -> KvChecksumComputer:
        return cls(
            kv_args.kv_data_ptrs,
            kv_args.kv_item_lens,
            kv_args.page_size,
            kv_args.state_types,
            kv_args.state_data_ptrs,
            kv_args.state_item_lens,
        )

    def _layout_signature(self) -> int:
        """Identify the layout a digest is taken over.

        Two sides can compare digests only if they cover the same bytes. A
        prefill with another TP width has other per-page item lengths; a PP or
        layer-sharded prefill owns fewer buffers. Shipping this beside the
        digest turns those into one logged skip instead of a mismatch on every
        request.
        """
        layout = (
            self.page_size,
            self._kv_item_lens,
            [(state_type.value, lens) for state_type, _, lens in self._state],
        )
        raw = hashlib.blake2b(repr(layout).encode(), digest_size=8).digest()
        # Positive int64 and never 0, which marks "no digest" in the metadata row.
        return int.from_bytes(raw, "big") >> 1 | 1

    def compute(
        self,
        kv_pages: torch.Tensor,
        state_rows: Sequence[Optional[torch.Tensor]] = (),
    ) -> int:
        """`state_rows[i]` addresses the i-th digested state component; None
        skips it."""
        ptrs, lens = list(self._kv_data_ptrs), list(self._kv_item_lens)
        rows: List[torch.Tensor] = [kv_pages] * len(ptrs)
        for (_, component_ptrs, component_lens), idx in zip(self._state, state_rows):
            if idx is not None:
                ptrs += component_ptrs
                lens += component_lens
                rows += [idx] * len(component_ptrs)
        return adler32_strided_checksum(ptrs, lens, rows)

    def digest(self, scheduler, req: Req, seq_len: int) -> int:
        """Digest the handoff of `req` as this engine holds it."""
        allocator = scheduler.token_to_kv_pool_allocator
        page_size = allocator.page_size
        req_to_token = scheduler.req_to_token_pool.req_to_token
        row = req.kv.req_pool_idx
        # Page-aligned, and the same on both sides by the PD protocol.
        start = req.disagg_decode_prefix_len
        # As the send path does: a unified-memory pool keeps VIRTUAL ids in
        # req_to_token while the registered buffers take physical ones.
        kv_pages = _page_indices(
            allocator.translate_kv_indices_for_transfer(
                req_to_token[row, start:seq_len]
            ),
            page_size,
        )
        pool = allocator.get_kvcache()
        state_rows: List[Optional[torch.Tensor]] = []
        for state_type, _, _ in self._state:
            if state_type == StateType.MAMBA:
                req_pool = scheduler.req_to_token_pool
                idx = req_pool.translate_mamba_indices(
                    req_pool.req_index_to_mamba_index_mapping[row]
                )
                idx = idx.to(torch.int64).reshape(-1).contiguous()
            elif (
                state_type in _SWA_STATE_TYPES
                and isinstance(pool, SWAKVPool)
                and pool.request_window is None
            ):
                window_start = max(start, seq_len - scheduler.sliding_window_size)
                window_start -= window_start % page_size
                idx = _page_indices(
                    allocator.translate_swa_indices_for_transfer(
                        req_to_token[row, window_start:seq_len]
                    ),
                    page_size,
                )
            elif state_type == StateType.DSA:
                # The indexer cache is paged by the device pool, as on the
                # decode send path, which may differ from the allocator.
                idx = _page_indices(req_to_token[row, start:seq_len], pool.page_size)
            elif state_type in _PAGE_STATE_TYPES:
                idx = kv_pages
            else:
                idx = None
            state_rows.append(idx)
        return self.compute(kv_pages, state_rows)
