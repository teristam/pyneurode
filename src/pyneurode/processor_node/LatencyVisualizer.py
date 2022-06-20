# Visualize the latency of various processor
from .Visualizer import Visualizer
from typing import List
from pyneurode.processor_node.Message import Message
import dearpygui.dearpygui as dpg
from pyneurode.RingBuffer.RingBuffer import RingBuffer
import numpy as np
from .ProcessorContext import ProcessorContext
from .Processor import RandomMsgTimeSource
from .GUIProcessor import GUIProcessor
import random


class LatencyVisualizer(Visualizer):
    def __init__(self, name: str, subplot_size=(3, 3)) -> None:
        super().__init__(name)
        self.name = name  # unique identying string for the visualizer
        self.data_count = 0
        self.plot_data_tag = self.name + "_plot_data"
        self.buffer = None
        self.first_run = True
        self.time_cursor = "time_cursor"
        self.data_idx = None
        self.plot_data = (
            {}
        )  # each item should be a tuple with x and y data (xdata,ydata,data_count)
        self.plot_tags = []
        self.subplot_size = subplot_size
        self.buffer_length = 100

    def init_gui(self):
        with dpg.window(label=self.name, width=300, height=200, tag=self.name):
            with dpg.subplots(
                self.subplot_size[0], self.subplot_size[1], width=-1, height=-1
            ) as self.subplots:
                # dpg.add_plot_legend()
                for metrics in self.plot_data.keys():
                    with dpg.plot() as plot_id:
                        dpg.add_plot_axis(dpg.mvXAxis, label="")
                        with dpg.plot_axis(dpg.mvYAxis, label="ms"):
                            dpg.add_line_series(
                                self.plot_data[metrics][0],
                                self.plot_data[metrics][1],
                                label=metrics,
                            )

    def update_buffer(self, buffer, x):
        # update the display buffer with the latest dat
        idx = buffer[2]
        buffer[1][idx % self.buffer_length] = x
        buffer[2] = (idx + 1) % self.buffer_length  # increment the index

    def update(self, messages: List[Message]):

        """
        1. check if the metrics is already in the current dictionary
        2. if not, add it to the existing dictionary, update the x axis tick
        3. set the value accordingly

        """
        for m in messages:

            process_name = m.data['processor']
            measures = m.data['measures']

            for m, v in measures.items():
                metrics_name = process_name + ':' +  m
                if not metrics_name in self.plot_data:
                    # create a new subplot

                    # create the x and y data
                    self.plot_data[metrics_name] = [
                        np.arange(self.buffer_length).tolist(),
                        [0] * self.buffer_length,
                        0,
                    ]  # xdata, ydata, data_count

                    self.update_buffer(self.plot_data[metrics_name], v)

                    # add new subplot
                    with dpg.plot(parent=self.subplots,label=metrics_name):
                        dpg.add_plot_axis(dpg.mvXAxis, label="")
                        with dpg.plot_axis(dpg.mvYAxis, label=""):
                            dpg.add_line_series(
                                self.plot_data[metrics_name][0],
                                self.plot_data[metrics_name][1],
                                label=metrics_name,
                                tag=metrics_name
                            )
                else:
                    # update existing subplots
                    self.update_buffer(self.plot_data[metrics_name], v)
                    # print(self.plot_data[metrics_name])
                    dpg.set_value(
                        metrics_name,
                        (self.plot_data[metrics_name][0], self.plot_data[metrics_name][1]),
                    )


if __name__ == "__main__":

    with ProcessorContext() as ctx:

        msgs = []
        processors = ["p1", "p2", "p3"]

        for i in range(100):
            msgs.append(
                Message(
                    "metrics",
                    {
                        "processor": random.choice(processors),
                        "measures": {
                            "latency": random.randrange(0, 10),
                            "items_per_hit": random.randrange(10, 20),
                        },
                    },
                )
            )

        msgSource = RandomMsgTimeSource(0.01, msgs)
        latencyVis = LatencyVisualizer("latency")

        gui = GUIProcessor()
        gui.register_visualizer(latencyVis, filters=["metrics"])

        msgSource.connect(gui, "metrics")

        ctx.register_processors(msgSource, gui)

        ctx.start()
