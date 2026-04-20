# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

pyneurode is a real-time signal processing framework for neural recordings. It focuses on online spike sorting and neural signal decoding, primarily targeting the Open Ephys GUI. It supports parallel processing for hundreds of tetrode channels in real-time.

## Build and Install

The project uses `uv` for dependency management and has a Cython extension that must be compiled:

```bash
# Install dependencies
uv sync

# Build the Cython extension (spike_sorter_cy)
uv run python setup.py build_ext --inplace
```

The Cython extension `spike_sorter_cy` (in `src/pyneurode/spike_sorter_cy.pyx`) must be rebuilt after changes to `.pyx` files.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest test/test_ringBuffer.py

# Run a single test
uv run pytest test/test_ringBuffer.py::test_write
```

Note: `test/test_sorting.py` requires a sample data fixture at `test/sample_spikes`.

## Architecture

### Processor Pipeline Pattern

The core abstraction is a **directed graph of Processors** running as separate OS processes, communicating via `multiprocessing.Queue`-backed `Channel` objects.

Key classes in `src/pyneurode/processor_node/`:

- **`Processor`** – base class. Override `process(message)` to implement logic. Runs in its own process via `ProcessorContext`.
- **`Source`** – subclass of Processor with no input; calls `process()` in a loop to generate messages.
- **`TimeSource`** – Source that triggers at a fixed time interval.
- **`Sink`** – subclass with no output; only consumes messages.
- **`BatchProcessor`** – buffers messages and processes them in batches.
- **`GUIProcessor`** – BatchProcessor that drives a Dear PyGui GUI in the main thread; hosts `Visualizer` objects.
- **`Channel`** – wraps a `Queue` with optional dtype-based message filtering (`filters` parameter on `connect()`).
- **`Message`** – simple data wrapper with `dtype`, `data`, `timestamp`, and `source` fields. Always send `Message` subclasses between processors.
- **`ProcessorContext`** – manages processor lifecycle. Used as a context manager (`with ProcessorContext() as ctx:`); auto-registers processors created inside the `with` block and calls `ctx.start()` on `__exit__`.

Typical pipeline setup:

```python
with ProcessorContext() as ctx:
    source = FileReaderSource('data.pkl', interval=0.05)
    sorter = SpikeSortProcessor(interval=0.01)
    gui = GUIProcessor()

    source.connect(sorter, filters='spike')   # only pass 'spike' messages
    sorter.connect(gui, ['df_sort', 'metrics'])

    ctx.start()
```

### Message Flow and Filtering

`processor.connect(other, filters=None)` creates a `Channel`. If `filters` is set (string or list of strings), only messages whose `dtype` matches are forwarded. Messages must be `Message` instances; sending raw data raises `TypeError`.

### Spike Sorting Pipeline

1. **`ZmqSource`** or **`FileReaderSource`** reads raw neural data and emits `Message('spike', ...)` and `Message('adc_data', ...)`.
2. **`SpikeSortProcessor`** (BatchProcessor) accumulates spikes, runs PCA + isosplit5 clustering to build templates, then does template matching (Cython-accelerated: `template_match_all_electrodes_cy`). Outputs `SpikeTrainMessage` and `SortedSpikeMessage`.
3. **`SyncDataProcessor`** synchronizes spike trains with analog signals.
4. **`GUIProcessor`** renders results via registered `Visualizer` subclasses (e.g., `AnalogVisualizer`, `SpikeClusterVisualizer`).

### Cython Extension

`src/pyneurode/spike_sorter_cy.pyx` provides the performance-critical `template_match_all_electrodes_cy` and `align_spike_cy` functions. It is compiled to `spike_sorter_cy.cp313-win_amd64.pyd` (Windows) and imported by `spike_sorter.py`.

### RingBuffer

`src/pyneurode/RingBuffer/` provides a circular buffer used to store spike train history for decoding. It supports timestamped reads (`readLatest`, `read` with time indices).

### Node Editor

`src/pyneurode/node_editor/node_editor.py` provides a Dear PyGui-based visual node editor (`NodeManager`) for interactively connecting processors at runtime. Entry point scripts like `sorter_node_editor.py` use this.

### Entry Point Scripts

The root-level `sorter_node*.py` scripts are standalone demos/experiments that wire up specific processor pipelines. `sorter_node.py` is the canonical example showing the full pipeline.

### ZMQ Interface

`ZmqSource` / `ZmqSubscriberSource` read live data from Open Ephys via ZMQ. `ZmqPublisherSink` / `zmq_client.py` handle the ZMQ protocol for the Open Ephys spike-sort plugin.
