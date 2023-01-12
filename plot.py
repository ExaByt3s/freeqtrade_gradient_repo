import math
import os

import pandas as pd
import numpy as np
import bokeh
from bokeh.models import (
    BooleanFilter,
    CDSView,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    HoverTool,
    NumeralTickFormatter,
)
from bokeh.plotting import figure

def plot(
    title,
    data,
    date: str = 'date',
    open: str = 'open',
    high: str = 'high',
    low: str = 'low',
    close: str = 'close',
    volume: str = 'volume',
    kind='candlestick',
    # kind='line',
    show_volume = True,
    addplot = None,
    # subplot=None,
    subplot = [],
    main_plot_height = 400,
    volume_plot_height = 100,
    crosshair_line_color = 'gray',
    crosshair_line_alpha: float = 0.50,
    bar_width: float = 0.75,
    theme: str = 'dark_minimal',
    grid_line_alpha: float = 0.25,
    filename: str = 'a.html',
    open_browser: bool = False,
):
    _date = date
    _open = open
    _high = high
    _low = low
    _close = close
    _volume = volume
    _kind = kind
    _show_volume = show_volume
    _addplot = addplot
    _main_plot_height = main_plot_height
    _volume_plot_height = volume_plot_height
    option_crosshair = dict(line_color=crosshair_line_color, line_alpha=crosshair_line_alpha)
    _linked_crosshair = CrosshairTool(dimensions='height', **option_crosshair)  # width, height, both  # https://docs.bokeh.org/en/latest/docs/reference/core/enums.html#bokeh.core.enums.Dimensions
    _p = []

    bokeh.plotting.curdoc().theme = theme
    INDEX_COL = 'index1'

    data[INDEX_COL] = data.index
    _source = ColumnDataSource(data)
    inc = _source.data[_close] >= _source.data[_open]
    dec = ~inc
    _view_inc = CDSView(source=_source, filters=[BooleanFilter(inc)])
    _view_dec = CDSView(source=_source, filters=[BooleanFilter(dec)])
    # _view = CDSView(source=_source)
    _options = dict(x_axis_type='datetime', tools='pan,xwheel_zoom,box_zoom,zoom_in,zoom_out,reset,save')  # plot_width=1000, sizing_mode='stretch_width' scale_height scale_both stretch_height scale_width , sizing_mode='scale_both'

    # https://www.tradingview.com/pine-script-docs/en/v5/concepts/Colors.html
    color_green = '#4CAF50'
    color_red = '#FF5252'

    color_increase = color_green
    color_decrease = color_red

    def _format_style(plot, **kwargs):
        styles = {}
        if plot == 'line':
            styles['color'] = kwargs['color'] if 'color' in kwargs else 'gray'
            styles['line_width'] = kwargs['line_width'] if 'line_width' in kwargs else 1
            styles['alpha'] = kwargs['alpha'] if 'alpha' in kwargs else 1
        elif plot == 'scatter':
            styles['color'] = kwargs['color'] if 'color' in kwargs else 'gray'
            styles['size'] = kwargs['size'] if 'size' in kwargs else 10
            styles['alpha'] = kwargs['alpha'] if 'alpha' in kwargs else 1
            styles['marker'] = kwargs['marker'] if 'marker' in kwargs else 'dot'

        return styles

    def _auto_scale(p):
        custom_js_args = dict(ohlc_range=p.y_range, source=_source)
        p.x_range.js_on_change('end', CustomJS(args=custom_js_args, code='''
        // Credit: https://github.com/kernc/backtesting.py
        // https://github.com/kernc/backtesting.py/blob/master/backtesting/autoscale_cb.js

        if (!window._bt_scale_range) {
          window._bt_scale_range = function (range, min, max, pad) {
            "use strict";
            if (min !== Infinity && max !== -Infinity) {
              pad = pad ? (max - min) * 0.03 : 0;
              range.start = min - pad;
              range.end = max + pad;
            } else console.error("backtesting: scale range error:", min, max, range);
          };
        }

        clearTimeout(window._bt_autoscale_timeout);

        window._bt_autoscale_timeout = setTimeout(function () {
          /**
           * @variable cb_obj `fig_ohlc.x_range`.
           * @variable source `ColumnDataSource`
           * @variable ohlc_range `fig_ohlc.y_range`.
           * @variable volume_range `fig_volume.y_range`.
           */
          "use strict";

          let i = Math.max(Math.floor(cb_obj.start), 0),
            j = Math.min(Math.ceil(cb_obj.end), source.data["High"].length);

          let max = Math.max.apply(null, source.data["High"].slice(i, j)),
            min = Math.min.apply(null, source.data["Low"].slice(i, j));
          _bt_scale_range(ohlc_range, min, max, true);

          // if (volume_range) {
            // max = Math.max.apply(null, source.data["Volume"].slice(i, j));
            // _bt_scale_range(volume_range, 0, max * 1.03, false);
          // }
        }, 50);
        '''))
        return p

    def _auto_scale_y(p):
        custom_js_args = dict(ohlc_range=p.y_range, source=_source)

        # https://github.com/ndepaola/bokeh-candlestick/blob/master/candlestick_plot.py
        # source = ColumnDataSource({'Index': df.index, 'High': df.High, 'Low': df.Low})
        # callback = CustomJS(args={'y_range': fig.y_range, 'source': source}, code='''
        callback = CustomJS(args=custom_js_args, code='''
            clearTimeout(window._autoscale_timeout);
            var Index = source.data.Index,
                Low = source.data.Low,
                High = source.data.High,
                start = cb_obj.start,
                end = cb_obj.end,
                min = Infinity,
                max = -Infinity;
            for (var i=0; i < Index.length; ++i) {
                if (start <= Index[i] && Index[i] <= end) {
                    max = Math.max(High[i], max);
                    min = Math.min(Low[i], min);
                }
            }
            var pad = (max - min) * .05;
            window._autoscale_timeout = setTimeout(function() {
                y_range.start = min - pad;
                y_range.end = max + pad;
            });
        ''')

        # Finalise the figure
        # p.x_range.callback = callback
        # p.x_range.js_on_change('end', callback)
        p.x_range.js_event_callbacks = callback
        return p

    def add_indicator(p, indicator_all):
        for indicator in indicator_all:
            indicator_object = None
            if indicator['kind'] == 'line':
                indicator_object = p.line(x=INDEX_COL, y=indicator['column'], source=_source,
                                          **_format_style('line', **indicator))
            elif indicator['kind'] == 'scatter':
                indicator_object = p.scatter(x=INDEX_COL, y=indicator['column'], source=_source,
                                             **_format_style('scatter', **indicator))
            else:
                raise ValueError('Other kinds are not supported.')

            p.add_tools(
                HoverTool(
                    renderers=[
                        indicator_object,
                    ],
                    tooltips=[
                        (indicator['column'], f"@{indicator['column']}" + '{0,0.00}'),
                    ],
                ),
            )

        return p

    def _candlestick_plot():
        p = figure(**_options)  # plot_height=_main_plot_height, , aspect_ratio=(16 / 9 / 0.25)

        _segment_option = dict(x0=INDEX_COL, x1=INDEX_COL, y0=_low, y1=_high, source=_source, line_width=2)
        s1 = p.segment(color=color_increase, view=_view_inc, **_segment_option)
        s2 = p.segment(color=color_decrease, view=_view_dec, **_segment_option)

        vbar_options = dict(x=INDEX_COL, width=bar_width, top=_open, bottom=_close, source=_source)
        t1 = p.vbar(fill_color=color_increase, line_color=color_increase, view=_view_inc, **vbar_options)
        t2 = p.vbar(fill_color=color_decrease, line_color=color_decrease, view=_view_dec, **vbar_options)

        p.add_tools(
            HoverTool(
                renderers=[t1, t2, s1, s2],
                tooltips=[
                    (_open, f'@{_open}' + '{0,0.00}'),
                    (_high, f'@{_high}' + '{0,0.00}'),
                    (_low, f'@{_low}' + '{0,0.00}'),
                    (_close, f'@{_close}' + '{0,0.00}'),
                ],
            ),
        )

        p = add_indicator(p, _addplot)

        p.add_tools(
            _linked_crosshair,
            CrosshairTool(dimensions='width', **option_crosshair),
        )

        _auto_scale(p)
        _p.append(p)

    def _line_plot():
        p = figure(**_options)  # plot_height=_main_plot_height, , aspect_ratio=(16 / 9 / 0.25)

        l = p.line(x=INDEX_COL, y=_close, source=_source)

        p.add_tools(
            HoverTool(
                renderers=[l],
                tooltips=[
                    (_close, f'@{_close}' + '{0,0.00}'),
                ],
            ),
        )

        p = add_indicator(p, _addplot)

        p.add_tools(
            _linked_crosshair,
            CrosshairTool(dimensions='width', **option_crosshair),
        )

        _auto_scale(p)
        _p.append(p)

    def _plot():
        if _kind == 'candlestick':
            _candlestick_plot()
        elif _kind == 'line':
            _line_plot()
        else:
            raise ValueError('Please choose from the following: candletitle, line')

    def _volume_plot():
        p = figure(x_range=_p[0].x_range, **_options)  # , plot_height=_volume_plot_height, aspect_ratio=(16 / 9 / 0.25)

        vbar_options = dict(
            x=INDEX_COL,
            width=bar_width,
            top=_volume,
            bottom=0,
            source=_source,
        )

        t1 = p.vbar(fill_color=color_increase, line_color=color_increase, view=_view_inc, **vbar_options)
        t2 = p.vbar(fill_color=color_decrease, line_color=color_decrease, view=_view_dec, **vbar_options)

        p.add_tools(
            HoverTool(
                renderers=[t1, t2],
                tooltips=[
                    (_volume, f'@{_volume}' + '{0,0.0[0000]}'),
                ],
            ),
            _linked_crosshair,
            CrosshairTool(dimensions='width', **option_crosshair),
        )

        p.yaxis.formatter = NumeralTickFormatter(format='0.0a')
        _p.append(p)

    def _add_subplot(subplot):
        p = figure(x_range=_p[0].x_range, **_options)  # plot_height=200, , aspect_ratio=(16 / 9 / 0.2)
        p = add_indicator(p, subplot)

        p.add_tools(
            _linked_crosshair,
            CrosshairTool(dimensions='width', **option_crosshair),
        )

        _p.append(p)

    def _set_xaxis():
        for p in _p:
            p.xaxis.visible = False

        p_last = _p[-1]
        p_last.xaxis.visible = True

        if False:
            p_last.xaxis.major_label_orientation = math.pi / 4

        # https://github.com/ndepaola/bokeh-candlestick/blob/master/candlestick_plot.py
        xaxis_dt_format = '%Y-%m-%d'
        if _source.data[_date][1] - _source.data[_date][0] < np.timedelta64(1, 'h'):
            xaxis_dt_format = '%Y-%m-%d %H:%M:%S'

        _major_label_overrides = {
            i: date.strftime(xaxis_dt_format)
            for i, date in enumerate(pd.to_datetime(_source.data[_date]))
        }

        p_last.xaxis.major_label_overrides = _major_label_overrides

        # https://qiita.com/saliton/items/6249d92f266ef435a5f0
        # p_last.xaxis.bounds = (-1, df.index[-1] + 1)
        # p_last.xaxis.bounds = (_source.data[INDEX_COL][0], _source.data[INDEX_COL][-1])
        p_last.x_range.range_padding = 0.01

    def _set_grid_line():
        for p in _p:
            p.grid.grid_line_alpha = grid_line_alpha

    def _set_title():
        _p[0].title = title

    def _object_grid():
        # p = bokeh.layouts.layout(children=[[_p]], sizing_mode='stretch_both')
        return bokeh.layouts.gridplot(children=_p, sizing_mode='stretch_both', toolbar_location='right', ncols=1,  # stretch_width scale_both stretch_both
                                      toolbar_options=dict(logo=None), merge_tools=True)

    _plot()

    if _show_volume:
        _volume_plot()

    for i in subplot:
        _add_subplot(i)

    _set_xaxis()
    _set_grid_line()
    _set_title()

    filepath = os.path.dirname(filename)

    if filepath != '':
        os.makedirs(filepath, exist_ok=True)

    bokeh.io.output_file(filename=filename, title=title, mode='inline')

    p = _object_grid()

    if open_browser:
        bokeh.io.show(obj=p)
    else:
        bokeh.io.save(obj=p)
