"""
Central configuration for the GRAND empirical pipeline.

Inputs: edit ``project_path`` to point at the local data root (raw market data
and all intermediate artifacts live under this directory).
Operations: expose shared constants used by preprocessing, training, and
inference scripts.
Outputs: module-level constants — data root, sample window
(``start_time`` / ``valid_time`` / ``end_time``), forecast ``horizon``,
lag order ``lag`` (feature lookback ``S``), and feature dimension ``P``.
"""

project_path = None
if project_path is None:
    raise ValueError('project_path is not set')

start_time = '2010-01-01'
valid_time = '2018-12-31'
end_time = '2022-07-31'
horizon = 1
lag = 48
P = 31