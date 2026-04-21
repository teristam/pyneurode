import dearpygui.dearpygui as dpg
import numpy as np 

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

with dpg.window(label="Histogram APP"):
    dpg.add_colormap_slider(label="Colormap Slider 1", default_value=0.5)
    cmap = dpg.bind_colormap(dpg.last_item(), dpg.mvPlotColormap_Spectral)
    
    with dpg.plot(label="Plot") as plot:
        # dpg.add_colormap_scale(label='color_map', min_scale=-1, max_scale=1)
        # dpg.bind_colormap(dpg.last_item(), dpg.mvPlotColormap_Spectral)

        dpg.bind_colormap(plot, cmap)
        
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis" )
        xdata = np.random.normal(0.5, scale=2, size=(1000,))
        ydata = np.random.normal(0.7, scale = 2, size=(1000,))

        dpg.add_2d_histogram_series(xdata, ydata, label='Histogram' , xbins=100, ybins=100, xmin_range=-1, xmax_range=1, ymin_range=-1, ymax_range=1,
                                    parent="x_axis")



dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()