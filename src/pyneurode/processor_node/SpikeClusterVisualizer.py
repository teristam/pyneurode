import logging
from typing import *

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from sklearn import cluster
from pyneurode.RingBuffer.RingBuffer import RingBuffer
from pyneurode.processor_node.Message import Message
from pyneurode.processor_node.MountainsortTemplateProcessor import RecomputeTemplateControlMessage
from .Visualizer import Visualizer
import time


def getClusterColor(i:int,alpha=200):
    colors =[
        (127,201,127,alpha),
        (190,174,212,alpha),
        (253,192,134,alpha),
        (255,255,153,alpha),
        (56,108,176,alpha),
        (240,2,127,alpha),
        (191,91,23,alpha)]

    return colors[i%len(colors)]


class SpikeClusterVisualizer(Visualizer):


    def __init__(self,num_channel=4, max_spikes = 500, max_plot_per_cluster=80) -> None:
        super().__init__()
        self.num_channel = num_channel
        self.channel_plot = []
        self.channel_plot_axes = []

        self.cluster_list = set()
        self.df_sort = pd.DataFrame() #only keep a fixed number of spikes
        self.max_spike = max_spikes
        self.max_plot_per_cluster = max_plot_per_cluster
        self.plot_need_refit = True # whether it is the first time the plots are shown
        self.template_update_id = None

    def init_gui(self):
        
        with dpg.window(label='Waveform window', autosize=True):
            window_width = 800
            window_height = window_width//2

            # PCA plot
            with dpg.group(horizontal=True):
                with dpg.plot(label = 'Principle Components', width = window_width//2, height=window_height):
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                    dpg.set_axis_limits_auto(x_axis)
                    dpg.set_axis_limits_auto(y_axis)
                    self.pca_axes = [x_axis, y_axis]

                # waveform plots
                with dpg.group(horizontal=True):
                    with dpg.group(label='Group1'):
                        for i in range(self.num_channel):
                            if i%2 == 0:
                                group = dpg.add_group(horizontal=True)

                            with dpg.plot(label = f'channel {i}',width = window_width//4, height=window_height//2, parent=group): # if -1, will scale with window size
                                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="")
                                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="")
                                dpg.set_axis_limits_auto(x_axis)
                                dpg.set_axis_limits_auto(y_axis)
                                

                                self.channel_plot_axes.append((x_axis, y_axis))

                    # Cluster select list  
                    with dpg.group(label='Selection'):
                        def refit_callback():
                            if dpg.get_value(self.checkbox_autofit): 
                                self.plot_need_refit = True

                        self.list_box=dpg.add_listbox(label='Clusters',items=list(self.cluster_list), width=-1,
                                callback=refit_callback)
                        
                        def btn_recompute_template(sender, app_data, user_data):
                            self.send_control_msg(RecomputeTemplateControlMessage())

                        # plotting options
                        self.checkbox_autofit = dpg.add_checkbox(label='Auto scale', default_value=True, callback=refit_callback)
                        
                                        
                        dpg.add_button(label='Re-compute template', callback=btn_recompute_template)

            self.init_theme()

    
    def auto_fit_plots(self):
        for ax,ay in self.channel_plot_axes:
            dpg.fit_axis_data(ax)
            dpg.fit_axis_data(ay)

        dpg.fit_axis_data(self.pca_axes[0])
        dpg.fit_axis_data(self.pca_axes[1])

    def init_theme(self):
        # generate a set of themes
        self.pca_themes = []

        marker_style = [
            dpg.mvPlotMarker_Diamond,
            dpg.mvPlotMarker_Asterisk,
            dpg.mvPlotMarker_Circle,
            dpg.mvPlotMarker_Cross,
            dpg.mvPlotMarker_Cross
        ]

        for i in range(7):
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries, parent=theme):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, getClusterColor(i), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, marker_style[i%len(marker_style)], category=dpg.mvThemeCat_Plots)
                
                with dpg.theme_component(dpg.mvLineSeries, parent=theme):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, getClusterColor(i), category=dpg.mvThemeCat_Plots)

                self.pca_themes.append(theme)

    def drawSpikeWaveform(self):
        #TODO: add button to auto fit all graphs
        
        # Plot the waveform
        sel_cluster_id = int(dpg.get_value(self.list_box))
        df_sel = self.df_sort[self.df_sort.cluster_id == int(sel_cluster_id)]
        if len(df_sel)>0:
            spike_waveforms = df_sel.spike_waveform_aligned.values
            xdata = np.arange(len(spike_waveforms[0])//self.num_channel).tolist()
            spike_length = len(spike_waveforms[0])//self.num_channel
            
            for i in range(min(len(spike_waveforms),self.max_plot_per_cluster)): # limit the amount of waveform to plot
                for j in range(self.num_channel):
                    ydata = spike_waveforms[i][j*spike_length:(j+1)*spike_length].tolist()
                    series = dpg.add_line_series(xdata, ydata, parent=self.channel_plot_axes[j][1])
                    dpg.bind_item_theme(series, self.get_cluster_theme(sel_cluster_id)) # draw with the same color as in the PCA plot

    def get_cluster_theme(self, cluster2plot):
        # get the theme for a specific cluster_id

        # first get all the clusters in the same tetrode
        tetrode = cluster2plot//100 
        cur_tetrode_cluster = [c for c in self.cluster_list if c//100==tetrode] #find the cluster id for current tetrode
        
        # find the proper index in the theme
        idx = cur_tetrode_cluster.index(cluster2plot)
        return self.pca_themes[idx%len(self.pca_themes)]

    def drawPCA(self):
        cluster2plot = dpg.get_value(self.list_box) #will return a string
        tetrode = int(cluster2plot)//100 # cluster id is in the form 201, 202 etc, where n%100 is the channel
        cur_tetrode_cluster = [c for c in self.cluster_list if (c//100==tetrode) and c>0] #find the cluster id for current tetrode, skip the noise class

        for i,cid in enumerate(cur_tetrode_cluster):
            df_sel = self.df_sort[self.df_sort.cluster_id == cid]

            if len(df_sel)>0:
                if 'pc_norm' in df_sel.columns:
                    # print(df_sel[['pc_norm','cluster_id']])
                    pc_norm = np.stack(df_sel.pc_norm.to_numpy())
                    if pc_norm[0] is not None:
                        x = pc_norm[:,0].tolist()
                        y = pc_norm[:,1].tolist()

                        series = dpg.add_scatter_series(x, y, parent=self.pca_axes[1])
                        # apply theme to the series
                        dpg.bind_item_theme(series, self.get_cluster_theme(cid))

    def clear_display(self):
        self.log(logging.DEBUG, 'Clearing display')
        self.df_sort = pd.DataFrame()
        self.cluster_list = set()


    def update(self, messages: List[Message]):
        start = time.time()
        # clear the plot
        for ax,ay in self.channel_plot_axes:
            dpg.delete_item(ay, children_only=True)

        dpg.delete_item(self.pca_axes[1], children_only=True)

        # Note: when there are lots of message, this part will be super slow, may make more sense to discard some of the data
        # since it is only for visualization
        
        
        df = pd.concat([m.data for m in  messages])
        
        # clear current wavefroms if it is using a new template
        if self.template_update_id is None:
            self.template_update_id = df.iloc[-1].template_update_id
        elif not self.template_update_id == df.iloc[-1].template_update_id:
            self.clear_display()
            self.template_update_id = df.iloc[-1].template_update_id
        
        # avoid appending large dataframe
        if len(df) > self.max_spike:
            # too many, only get the latest spikes
            self.df_sort = df[-self.max_spike:-1]
        else:
            # combine the correct portion
            leftover_length = self.max_spike - len(self.df_sort)
            self.df_sort = pd.concat([self.df_sort[-leftover_length:], df], ignore_index=True)


        if len(self.df_sort)>0:
            #update the cluster id
            cluster_ids = set(self.df_sort.cluster_id.unique())
            # print('cluster id in visuzlier: ', self.df_sort.cluster_id)
            if not cluster_ids in self.cluster_list:
                self.cluster_list = self.cluster_list.union(cluster_ids)
                items = sorted(list(self.cluster_list))
                dpg.configure_item(self.list_box, items=items, num_items=len(items))

            self.drawSpikeWaveform()
            self.drawPCA()

            # auto fit the plots on first plot
            if self.plot_need_refit:
                if dpg.get_value(self.checkbox_autofit):
                    self.auto_fit_plots()
                self.plot_need_refit = False

    
        lapsed = time.time()-start
        if lapsed > 0.1:
            print(f'warning: spike cluster udpate time: {lapsed}')
            print(f'mesage queue in spike visulizer: {len(messages)}')
            print(f'Total row: {sum([len(m.data) for m in messages])}')
