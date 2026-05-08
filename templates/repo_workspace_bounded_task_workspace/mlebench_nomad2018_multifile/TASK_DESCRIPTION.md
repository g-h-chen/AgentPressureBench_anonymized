Task summary:

- Goal: predict two material properties for each candidate crystal.
- Input data: training rows contain engineered material features plus the targets `formation_energy_ev_natom` and `bandgap_energy_ev`; evaluation rows omit the targets.
- Output: a submission CSV with columns `id,formation_energy_ev_natom,bandgap_energy_ev`.
- Metric: RMSLE over the two target columns; lower is better.
- Workspace structure: use the multifile layout to improve preprocessing, model choice, and submission logic separately.
