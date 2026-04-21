import logging
import time
from typing import List

import numpy as np
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.processor_node.AnalogVisualizer import AnalogVisualizer, DummpyControlSink
from pyneurode.processor_node.GUIProcessor import GUIProcessor
from pyneurode.processor_node.Message import Message, NumpyMessage
from pyneurode.processor_node.Processor import MsgTimeSource
from pyneurode.processor_node.SineTimeSource import SineTimeSource
from pyneurode.processor_node.ProcessorContext import ProcessorContext
import dearpygui.dearpygui as dpg

class TuningCurveEstimator():
    def __init__(self, nbin, bin_size=1):
        self.tuning_curve = None
        self.bin_size = bin_size
        self.bin_n = np.zeros((nbin,))
        self.nbin = nbin
        
    def update(self, x, y):
        """_summary_

        Args:
            x (np.array): behavioral variable
            y (np.array): (time x channel)
        """
        x = x//self.bin_size
        
        # assert np.max(x) < self.nbin, 'The large data exceed the number of bin'
        assert len(x) == len(y), 'Length of two inputs must be the same'
        
        if self.tuning_curve is None:
            self.tuning_curve = np.zeros((y.shape[1], self.nbin))
            for i in range(self.nbin):
                ysub = y[i==x,:]
                if len(ysub)>0:
                    self.tuning_curve[:,i] = np.mean(ysub,axis=0)
                    self.bin_n[i] = np.sum(i==x)
        else:
            # update the current mean
            for i in range(self.nbin):
                m = np.sum(i==x)
                if m>0:
                    self.tuning_curve[:,i] = (self.bin_n[i]*self.tuning_curve[:,i] + m*np.mean(y[i==x,:],axis=0))/(m + self.bin_n[i])
                    self.bin_n[i] += m
    
    def get_tuning_curve(self):
        return (np.arange(self.nbin), self.tuning_curve)
    
    def get_num_cell(self):
        if self.tuning_curve is not None:
            return self.tuning_curve.shape[0]
        else:
            return 0


class TuningCurveVisualizer(AnalogVisualizer):
    """
    A visualizer that calculate and display the 1D tuning curve of cells in realtime
    """
    def __init__(self, x_variable_idx=0, filters=None, nbin=10, bin_size: float = 1,
                 buffer_length:int = 500, scale = 1, time_scale=None, smoothing_alpha=0.1, title=None) -> None:
        super().__init__(filters=filters, title=title)
        self.plot_data = (np.arange(buffer_length), np.zeros((buffer_length,)))
        self.data_count = 0
        self.plot_data_tag = self.name+'_plot_data'
        self.buffer = None
        self.buffer_length = buffer_length
        self.need_fit_axis = True
        self.scale = scale
        self.time_scale = time_scale # time period of each data point
        self.time_cursor = 'time_cursor'
        self.data_idx = None
        self.series_name = []
        self.control_list = []
        self.y_axes: dict[int, int | str] = {}
        self.x_axes: dict[int, int | str] = {}
        self.selected_cells = []
        self.subplots = None
        self._needs_subplot_rebuild = False
        self.tuning_curve_estimator = TuningCurveEstimator(nbin = nbin, bin_size=bin_size)
        self.x_variable_idx = x_variable_idx
        
        
    def init_gui(self):
        window_width = 800
        with dpg.window(label=(self.title if self.title else self.name), width=window_width, height=500, tag=self.name, autosize=True):
            with dpg.group(horizontal=True):
                self.plot_panel = dpg.add_child_window(width=-window_width*1/4) #negative means no. of pixel from the right of its container
                self.control_panel = dpg.add_child_window(width=window_width*1/4)
                dpg.add_spacer(height = 20, parent=self.control_panel)
                    
    def update_cell_select_list(self):
        # update the checkboxes for selecting which cell to trigger
        no_cell = self.tuning_curve_estimator.get_num_cell()
        old_selected = self.selected_cells
        self.selected_cells = [
            old_selected[i] if i < len(old_selected) else False
            for i in range(no_cell)
        ]
        self.series_name = [f'{i}_{self.name}' for i in range(len(self.selected_cells))]
        self.y_axes = {}
        self.x_axes = {}

        # clear all the existing cells
        for gp in self.control_list:
            dpg.delete_item(gp)

        self.control_list = []

        def selected_cell_changed(sender, is_selected, cell_no):
            try:
                if cell_no >= len(self.selected_cells):
                    logging.error(
                        f'Checkbox callback cell_no={cell_no} is out of range '
                        f'(selected_cells has {len(self.selected_cells)} entries). '
                        f'This usually means the cell list was rebuilt while a stale checkbox still exists.'
                    )
                    return
                self.selected_cells[cell_no] = is_selected
                self._needs_subplot_rebuild = True
            except Exception as e:
                logging.error(
                    f'selected_cell_changed crashed: cell_no={cell_no}, '
                    f'is_selected={is_selected}, '
                    f'len(selected_cells)={len(self.selected_cells)}, '
                    f'series_name={self.series_name}, '
                    f'error={e}',
                    exc_info=True
                )

        for i in range(no_cell):
            with dpg.group(horizontal=True,parent=self.control_panel) as gp:
                dpg.add_checkbox(label = f'Channel {i}', callback=selected_cell_changed, user_data=i)
                self.control_list.append(gp)

        # clear stale subplots now that cell count has changed
        self._needs_subplot_rebuild = True
                
                
    def update_subplots(self):
        try:
            if self.subplots is not None:
                dpg.delete_item(self.subplots)
                self.subplots = None
            self.y_axes = {}
            self.x_axes = {}

            self.log(logging.DEBUG, self.series_name)

            ncol = 3
            n_selected = sum(self.selected_cells)
            if n_selected == 0:
                self.subplots = None
                return
            ncol = min(ncol, n_selected)
            nrow = int(np.ceil(n_selected / ncol))

            if len(self.series_name) != len(self.selected_cells):
                logging.error(
                    f'update_subplots: series_name length ({len(self.series_name)}) '
                    f'does not match selected_cells length ({len(self.selected_cells)}). '
                    f'State is inconsistent — skipping subplot rebuild.'
                )
                return

            plots_created = 0
            with dpg.subplots(nrow, ncol, parent=self.plot_panel, width=-1, height=-1) as subplots:
                self.subplots = subplots
                for i, is_selected in enumerate(self.selected_cells):
                    if is_selected:
                        tag = self.series_name[i]
                        if dpg.does_item_exist(tag):
                            logging.warning(
                                f'update_subplots: tag "{tag}" already exists in DPG before creation. '
                                f'Deleting stale item to avoid duplicate-tag crash.'
                            )
                            dpg.delete_item(tag)
                        with dpg.plot():
                            dpg.add_plot_legend()
                            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                            self.y_axes[i] = yaxis
                            self.x_axes[i] = xaxis
                            dpg.add_line_series([], [], label=tag, parent=yaxis, tag=tag)
                            dpg.fit_axis_data(self.y_axes[i])
                            dpg.fit_axis_data(self.x_axes[i])
                        plots_created += 1

            self.need_fit_axis = True
        except Exception as e:
            logging.error(
                f'update_subplots crashed: '
                f'n_selected={sum(self.selected_cells)}, '
                f'selected_cells={self.selected_cells}, '
                f'series_name={self.series_name}, '
                f'subplots_tag={self.subplots}, '
                f'error={e}',
                exc_info=True
            )

    def update(self, messages: List[Message]):

        if self._needs_subplot_rebuild:
            self._needs_subplot_rebuild = False
            self.update_subplots()

        messages = [m for m in messages if isinstance(m.data, np.ndarray)]
        for m in messages:
            if m.data.ndim != 2:
                continue
            n_cols = m.data.shape[1]
            if n_cols < 2:
                continue

            #plot y against x, where x is the variable for tuning curve (e.g. position) and y is the neural activity
            x_col = self.x_variable_idx % n_cols  # normalise negative index
            y_cols = [j for j in range(n_cols) if j != x_col]
            self.tuning_curve_estimator.update(m.data[:, x_col], m.data[:, y_cols])
            
        
        if not len(self.control_list) == self.tuning_curve_estimator.get_num_cell():
            self.update_cell_select_list()
        
        
        # update the plot
        
        x, y = self.tuning_curve_estimator.get_tuning_curve()
        # if y is None:
        #     return
        # for i, is_selected in enumerate(self.selected_cells):
        #     if is_selected:
        #         if i >= y.shape[0]:
        #             continue
        #         tag = self.series_name[i]
        #         if not dpg.does_item_exist(tag):
        #             continue
        #         dpg.set_value(tag, (x, y[i, :]))  # must be called in main thread
        
        
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    x = np.random.randint(0,20,(200,1))
    y = np.zeros((len(x),3),dtype=float)
    for i in range(len(x)):
        y[i,:] = np.random.normal(x[i]%10, size=3)
        
    data = np.hstack([y,x])
    print(data.shape) # [200,4]
    
    msg = []
    for _ in range(2000):
        idx = np.random.choice(np.arange(len(data)), size=100)
        msg += [NumpyMessage(data[idx,:])]

    with ProcessorContext(auto_start=False) as ctx:
        source = MsgTimeSource(1, msg)
        visualizer = TuningCurveVisualizer(x_variable_idx=-1, nbin=20,
                                           buffer_length=6000, title='Synchronized signals')
        control = DummpyControlSink()

        gui = GUIProcessor(internal_buffer_size=5000)

        gui.register_visualizer(source, [visualizer], control_targets=[control])

        ctx.start()
