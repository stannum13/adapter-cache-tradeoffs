from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_adapter: dict[str, str] = field(default_factory=dict)
    adapter_load: dict[str, int] = field(default_factory=dict)
    adapter_assignment_count: dict[str, int] = field(default_factory=dict)
    warm_adapters: set[str] = field(default_factory=set)
    adapter_tenants: dict[str, set[str]] = field(default_factory=dict)
    adapter_trust_groups: dict[str, set[str]] = field(default_factory=dict)

    def remember(
        self,
        session_id: str,
        adapter_id: str,
        tenant_id: str | None = None,
        trust_group_id: str | None = None,
    ) -> None:
        self.session_adapter[session_id] = adapter_id
        self.adapter_assignment_count[adapter_id] = (
            self.adapter_assignment_count.get(adapter_id, 0) + 1
        )
        self.warm_adapters.add(adapter_id)
        if tenant_id is not None:
            self.adapter_tenants.setdefault(adapter_id, set()).add(tenant_id)
        if trust_group_id is not None:
            self.adapter_trust_groups.setdefault(adapter_id, set()).add(trust_group_id)

    def begin_dispatch(self, adapter_id: str) -> None:
        self.adapter_load[adapter_id] = self.adapter_load.get(adapter_id, 0) + 1

    def end_dispatch(self, adapter_id: str) -> None:
        current = self.adapter_load.get(adapter_id, 0)
        if current <= 1:
            self.adapter_load.pop(adapter_id, None)
            return
        self.adapter_load[adapter_id] = current - 1

    def active_load(self, adapter_id: str) -> int:
        return self.adapter_load.get(adapter_id, 0)

    def isolation_penalty(
        self,
        adapter_id: str,
        tenant_id: str,
        trust_group_id: str,
        isolation_scope: str,
    ) -> float:
        if isolation_scope == "none":
            return 0.0
        if isolation_scope == "tenant":
            tenants = self.adapter_tenants.get(adapter_id, set())
            return 1.0 if tenants and tenant_id not in tenants else 0.0
        trust_groups = self.adapter_trust_groups.get(adapter_id, set())
        return 1.0 if trust_groups and trust_group_id not in trust_groups else 0.0
