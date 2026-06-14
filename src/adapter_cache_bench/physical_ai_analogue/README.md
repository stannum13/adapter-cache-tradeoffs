# Physical AI analogue

The LLM serving problem has a close analogue in physical AI and VLA serving.

| LLM serving | Physical AI / VLA serving |
| --- | --- |
| text prefix tokens | repeated visual/proprioceptive scene tokens |
| KV cache | visual/world-state cache |
| LoRA adapter | skill or embodiment adapter |
| adapter router | skill router |
| TTFT/goodput | control latency/success-rate-adjusted control Hz |

In both settings, specialization can improve task success while reducing reuse of expensive shared context. A skill router that always picks the semantically best skill can force repeated scene encoding under different skill namespaces. Late specialization and copy-on-write-style world-state deltas model the same design space as activated adapters for causal transformer serving.

This simulator has no robotics dependencies. It models static and dynamic scene tokens, skill adapters, scene-token cache reuse, action latency, success probability, and safety violation probability.

