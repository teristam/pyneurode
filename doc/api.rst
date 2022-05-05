API documentation
===================

Built-in Processor nodes
----------------------------

Each kind of processor node works on the data in its queue in a specific ways. 


Processor
^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.Processor.Processor
   :members: 

BatchProcessor
^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.BatchProcessor.BatchProcessor
   :members:
   :special-members: __init__



SpikeSortProcessor
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.SpikeSortProcessor.SpikeSortProcessor
   :members:
   :special-members: __init__


SyncDataProcessor
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.SyncDataProcessor.SyncDataProcessor
   :members:
   :special-members: __init__

GUIProcessor
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyneurode.processor_node.GUIProcessor.GUIProcessor
   :members:
   :special-members: __init__



Visualizers
------------------------

Visualizers are responsible for visualizing messages on the user interface using dearpygui

Visualizer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.Visualizer.Visualizer
   :members:
   :special-members: __init__

LatencyVisualizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.LatencyVisualizer.LatencyVisualizer
   :members:
   :special-members: __init__

AnalogVisualizer
^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.AnalogVisualizer.AnalogVisualizer
   :members:
   :special-members: __init__

SpikeClusterVisualizer
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyneurode.processor_node.SpikeClusterVisualizer.SpikeClusterVisualizer
   :members:
   :special-members: __init__
