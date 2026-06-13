from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_adapter: dict[str, str] = field(default_factory=dict)
    adapter_load: dict[str, int] = field(default_factory=dict)
    warm_adapters: set[str] = field(default_factory=set)

    def remember(self, session_id: str, adapter_id: str) -> None:
        self.session_adapter[session_id] = adapter_id
        self.adapter_load[adapter_id] = self.adapter_load.get(adapter_id, 0) + 1
        self.warm_adapters.add(adapter_id)
