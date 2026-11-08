"""L04 task family plugin package."""
from benchmark.plugins.task_family.l04.families import FAMILIES, FAMILY_BY_ID, Family
from benchmark.plugins.task_family.l04.plugin import L04TaskFamilies, PLUGIN

__all__ = ["FAMILIES", "FAMILY_BY_ID", "Family", "L04TaskFamilies", "PLUGIN"]
