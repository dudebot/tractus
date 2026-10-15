# Memory and throughput considerations

This historical filename is retained so older research links still resolve.
Development is stopped; this document does not authorize a migration or run.

The simulator differentiates through a per-sample MNA solve. Its important
memory/compute controls are frame_chunk_size, batch size, utterance duration,
and checkpoint policy. Profile each change on the target platform. More device
memory does not itself establish throughput, sound quality, or convergence.

- Smaller frame chunks and checkpointing trade recomputation for memory.
- Larger batches and longer sequences can increase peak memory; measure them.
- Keep fp32 as the default. fp64 remains an isolated parity oracle.
- Retain custom_vjp and custom_linear_solve correctness fixes.
- The backward memory gate remains a regression diagnostic.
- Require per-case convergence and acoustic validation before interpreting a run.

Earlier accelerator throughput projections were unmeasured. They are not
validated by this archive, and must not be used as capacity commitments.
