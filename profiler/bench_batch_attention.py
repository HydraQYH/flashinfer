import itertools
import json
import numpy as np
import torch
import flashinfer
from flashinfer.profiler import export_to_perfetto_trace

# Define constexpr
page_size = 1
num_kv_heads = 8
num_qo_heads = 32
head_dim = 128
kvcache_layout = "NHD"
test_dtype = torch.bfloat16
causal = True
flipped = False

def build_lens(prefill: int, prefill_prefix: int, decode_kv_len: int, p_batch: int = 1, d_batch: int = 128):
  kv_lens = [decode_kv_len for _ in range(d_batch)] + [prefill_prefix] * p_batch
  qo_lens = [1 for _ in range(d_batch)] + [prefill] * p_batch
  return kv_lens, qo_lens

def estimate_memory_footprint(kv_lens: list):
  np_kv_lens = np.asarray(kv_lens, dtype=np.int32)
  np_kv_lens_blocks = np.ceil(np_kv_lens, np.asarray([page_size for _ in range(len(kv_lens))], dtype=np.int32))
  num_blocks= np.sum(np_kv_lens_blocks)
  kv_cache = num_blocks * 2 * page_size * num_kv_heads * head_dim * 2 / 1024 / 1024 / 1024
  return kv_cache

def profile_persistent_batch_attention(
    kv_lens,
    qo_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    test_dtype,
    causal,
    tarce_file_prefix,
    profiler_buffer_size=2048576,
    device="cuda",
    flipped=False
):
  seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
  q_lens = torch.tensor(qo_lens, dtype=torch.int32)

  seq_lens_blocks = torch.ceil(seq_lens / page_size).int()

  q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
  kv_indptr = torch.cat(
    [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
  ).int()

  num_blocks = kv_indptr[-1].item()

  q = torch.rand(
    q_indptr[-1].item(), num_qo_heads, head_dim, device=device, dtype=test_dtype
  )
  if layout == "NHD":
    kv_data = torch.randn(
      num_blocks,
      2,
      page_size,
      num_kv_heads,
      head_dim,
      dtype=test_dtype,
      device=device,
    )
  elif layout == "HND":
    kv_data = torch.randn(
      num_blocks,
      2,
      num_kv_heads,
      page_size,
      head_dim,
      dtype=test_dtype,
      device=device,
    )

  wrapper = flashinfer.BatchAttention(kv_layout=layout)
  wrapper.plan(
    q_indptr.to(device),
    kv_indptr.to(device),
    torch.arange(num_blocks).int().to(device),
    seq_lens.to(device),
    num_qo_heads,
    num_kv_heads,
    head_dim,
    head_dim,
    page_size,
    causal=causal,
    q_data_type=test_dtype,
    kv_data_type=test_dtype,
    use_profiler=True,
    flipped_schedule=flipped,
  )

  profiler_buffer = torch.zeros(
    (profiler_buffer_size,), dtype=torch.uint64, device=device
  )

  # warmup
  start_event, end_event = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
  )
  wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
  profiler_buffer.zero_()

  torch.cuda.synchronize()
  start_event.record()
  wrapper.run(q, kv_data, profiler_buffer=profiler_buffer)
  end_event.record()
  end_event.synchronize()
  elapsed_time = start_event.elapsed_time(end_event)
  print(f"Kernel execution time: {start_event.elapsed_time(end_event)} ms")

  events = ["prefill", "decode", "reduction"]
  export_to_perfetto_trace(profiler_buffer, events, tarce_file_prefix)
  print(f"Profile trace exported to {tarce_file_prefix}")
  return elapsed_time

if __name__ == "__main__":
  new_prefill_tokens = list(range(512, 8192 + 1, 256))
  prefill_prefix_lens = list(range(0, 4096 + 1, 256))
  decode_kv_lens = list(range(512, 16384 + 1, 512))

  data = []

  for prefill, prefill_prefix, decode_kv_len in itertools.product(new_prefill_tokens, prefill_prefix_lens, decode_kv_lens):
    p_batch = 1
    d_batch = 128
    kv_lens, qo_lens = build_lens(prefill, prefill_prefix, decode_kv_len, p_batch=p_batch, d_batch=d_batch)
    kvcache_size = estimate_memory_footprint(kv_lens)
    print(f"KV Cache: {kvcache_size} GiB")
    if kvcache_size > 2.0:
      print("More than 2 GiB KVCache, skip.")
      continue
    trace_file_prefix = f"BatchAttention_pbatch_{p_batch}_p_{prefill}_pprefix_{prefill_prefix}_dbatch_{d_batch}_dkvlen_{decode_kv_len}_flipped_{flipped}.perfetto-trace"
    elapsed_time = profile_persistent_batch_attention(
      kv_lens,
      qo_lens,
      page_size,
      num_kv_heads,
      num_qo_heads,
      head_dim,
      kvcache_layout,
      test_dtype,
      causal,
      trace_file_prefix,
      flipped=flipped
    )

    if test_dtype == torch.bfloat16:
      test_dtype_str = "torch.bfloat16"
    elif test_dtype == torch.float16:
      test_dtype_str = "torch.float16"
    else:
      assert False

    data.append(
      {
        "prefill_batch": p_batch,
        "prefill_tokens": prefill,
        "prefill_prefix": prefill_prefix,
        "decode_batch": d_batch,
        "decode_kv_len": decode_kv_len,
        "page_size": page_size,
        "num_kv_heads": num_kv_heads,
        "num_qo_heads": num_qo_heads,
        "head_dim": head_dim,
        "kvcache_layout": kvcache_layout,
        "test_dtype": test_dtype_str,
        "causal": causal,
        "trace_file_name": trace_file_prefix,
        "flipped": flipped,
        "elapsed_time": elapsed_time
      }
    )

  with open('data.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
      json_line = json.dumps(item, ensure_ascii=False)
      f.write(json_line + '\n')
