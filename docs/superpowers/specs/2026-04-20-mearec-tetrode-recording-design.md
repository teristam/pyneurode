# MEArec Simulated Tetrode Recording — Design Spec

**Date:** 2026-04-20  
**Status:** Approved

## Goal

Generate a static simulated tetrode recording using MEArec and export it as a pickle file
compatible with `FileReaderSource` for replay in the pyneurode pipeline. Replaces manual
waveform templates in `SpikeGeneratorSource` with a realistic noise + spike model.

## Scope

- Jupyter notebook: `notebooks/generate_templates_and_recordings.ipynb`
- Output: `data/sim_tetrode.pkl` (or user-configured path)
- Spikes only (no ADC/analog data)
- No modifications to existing pyneurode code

## Architecture & Data Flow

```
MEArec templates.h5  ──►  gen_recordings()
                              │
                         RecordingExtractor
                         (n_channels × n_timepoints, int16)
                         + ground truth spike trains
                              │
                    ┌─────────┴─────────────┐
                    │ For each spike event:  │
                    │  - look up tetrode idx │
                    │  - slice 4-ch waveform │
                    │  - wrap in             │
                    │    OpenEphysSpikeEvent │
                    └─────────┬─────────────┘
                              │
                    pickle.dump() per event (sorted by timestamp)
                              │
                    data/sim_tetrode.pkl
                              │
                    FileReaderSource  ──► pipeline
```

## Notebook Structure

### Cell 1 — Configuration (all tunable parameters)

```python
n_tetrodes = 4
neurons_per_tetrode = [2, 3, 2, 3]   # list, one entry per tetrode
firing_rates_hz = [[20, 15], [10, 25, 30], [20, 10], [15, 20, 5]]  # Hz per neuron
duration = 60          # seconds
Fs = 30000             # Hz
noise_level = 10       # µV
n_samples = 130        # waveform snippet length (samples)
pre_samples = 30       # samples before spike peak
templates_file = "path/to/templates.h5"
output_file = "data/sim_tetrode.pkl"
```

### Cell 2 — Probe setup

Build a tetrode probe using `probeinterface`. Each tetrode is a 2×2 square grid with 25 µm
pitch. Tetrodes are stacked vertically 200 µm apart.

```
Tetrode 0:  ch0  ch1      (y=0 µm)
            ch2  ch3      (y=25 µm)

Tetrode 1:  ch4  ch5      (y=200 µm)
            ch6  ch7      (y=225 µm)
...
```

Total channels = `n_tetrodes × 4`.

Channel grouping: `tetrode_channels[i] = [4*i, 4*i+1, 4*i+2, 4*i+3]`

### Cell 1b — Templates download (optional helper)

If `templates_file` does not exist, the notebook provides a helper cell that downloads MEArec
example templates via `mr.download_templates()` or points to the MEArec GitHub release assets.

### Cell 3 — MEArec recording generation

```python
import MEArec as mr

# params dict for gen_recordings
rec_params = mr.get_default_recordings_params()
rec_params['recordings']['duration'] = duration
rec_params['recordings']['noise_level'] = noise_level
rec_params['recordings']['fs'] = Fs
rec_params['spiketrains']['n_exc'] = sum(neurons_per_tetrode)
rec_params['spiketrains']['n_inh'] = 0
# firing rates set per-unit via rec_params['spiketrains']['rates']

recgen = mr.gen_recordings(
    templates=templates_file,
    params=rec_params,
    probe=probe,
)
```

Template assignment: MEArec assigns templates to units automatically based on the probe.
Since templates are selected from the `.h5` file filtered to channels overlapping each
tetrode's spatial region, the peak channel naturally falls within the correct tetrode group.
If the templates file has broad coverage, an explicit post-hoc check verifies that each
unit's peak channel is in `tetrode_channels[tetrode_idx]`; mismatched units are reassigned
to the nearest tetrode.

### Cell 4 — Pickle export

1. Load recording traces as int16 array: shape `(n_channels, n_timepoints)`
2. Load ground-truth spike trains per unit (sample indices)
3. Build a list of `(timestamp, tetrode_idx, waveform)` tuples across all units, sorted by timestamp
4. For each entry:
   - `waveform = traces[tetrode_channels[tetrode_idx], t-pre_samples:t-pre_samples+n_samples]`  shape `(4, n_samples)`
   - Build `OpenEphysSpikeEvent` header: `{'n_channels': 4, 'n_samples': n_samples, 'electrode_id': tetrode_idx, 'sorted_id': 0, 'timestamp': t, 'channel': 0, 'threshold': 6, 'source': 115}`
   - `spk_evt = OpenEphysSpikeEvent(header, waveform)`
   - `pickle.dump({'spike': spk_evt, 'data_timestamp': t / Fs}, f)`

## Key Constraints

- Recording traces loaded once into memory as int16 (~800 MB for 4 tetrodes × 60s)
- Spikes within `pre_samples` of recording start or `post_samples` of end are skipped
- Output format exactly matches what `FileReaderSource` expects: dict with `spike` and `data_timestamp` keys

## Out of Scope

- ADC/analog data generation
- NEURON-based template generation (`gen_templates`)
- Real-time streaming from MEArec
- Any changes to existing pyneurode processors
