import ast
import datetime
import itertools
from itertools import groupby
import os
import re
import sys
import typing
import warnings

import matplotlib
from matplotlib import cm, colors, patches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from upsetplot import util
from upsetplot.reformat import _get_subset_mask, query

#################################################################
# ADAPTED CODE FROM UPSETPLOT - had a a bug
# https://upsetplot.readthedocs.io/en/stable/
# prevents ImportError on matplotlib versions >3.5.2
#################################################################
try:
    from matplotlib.tight_layout import get_renderer

    RENDERER_IMPORTED = True
except ImportError:
    RENDERER_IMPORTED = False


def _process_data(
    df,
    *,
    sort_by,
    sort_categories_by,
    subset_size,
    sum_over,
    min_subset_size=None,
    max_subset_size=None,
    max_subset_rank=None,
    min_degree=None,
    max_degree=None,
    reverse=False,
    include_empty_subsets=False,
):
    results = query(
        df,
        sort_by=sort_by,
        sort_categories_by=sort_categories_by,
        subset_size=subset_size,
        sum_over=sum_over,
        min_subset_size=min_subset_size,
        max_subset_size=max_subset_size,
        max_subset_rank=max_subset_rank,
        min_degree=min_degree,
        max_degree=max_degree,
        include_empty_subsets=include_empty_subsets,
    )

    df = results.data
    agg = results.subset_sizes

    # add '_bin' to df indicating index in agg
    # XXX: ugly!
    def _pack_binary(X):
        X = pd.DataFrame(X)
        # use objects if arbitrary precision integers are needed
        dtype = np.object_ if X.shape[1] > 62 else np.uint64
        out = pd.Series(0, index=X.index, dtype=dtype)
        for _, col in X.items():
            out *= 2
            out += col
        return out

    df_packed = _pack_binary(df.index.to_frame())
    data_packed = _pack_binary(agg.index.to_frame())
    df["_bin"] = pd.Series(df_packed).map(
        pd.Series(np.arange(len(data_packed))[:: -1 if reverse else 1], index=data_packed)
    )
    if reverse:
        agg = agg[::-1]

    return results.total, df, agg, results.category_totals


def _multiply_alpha(c, mult):
    r, g, b, a = colors.to_rgba(c)
    a *= mult
    return colors.to_hex((r, g, b, a), keep_alpha=True)


class _Transposed:
    """Wrap an object in order to transpose some plotting operations

    Attributes of obj will be mapped.
    Keyword arguments when calling obj will be mapped.

    The mapping is not recursive: callable attributes need to be _Transposed
    again.
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getattr__(self, key):
        return getattr(self.__obj, self._NAME_TRANSPOSE.get(key, key))

    def __call__(self, *args, **kwargs):
        return self.__obj(*args, **{self._NAME_TRANSPOSE.get(k, k): v for k, v in kwargs.items()})

    _NAME_TRANSPOSE = {
        "align_xlabels": "align_ylabels",
        "align_ylabels": "align_xlabels",
        "bar": "barh",
        "barh": "bar",
        "bottom": "left",
        "get_figheight": "get_figwidth",
        "get_figwidth": "get_figheight",
        "get_xlim": "get_ylim",
        "get_ylim": "get_xlim",
        "height": "width",
        "hlines": "vlines",
        "hspace": "wspace",
        "left": "bottom",
        "right": "top",
        "set_autoscalex_on": "set_autoscaley_on",
        "set_autoscaley_on": "set_autoscalex_on",
        "set_figheight": "set_figwidth",
        "set_figwidth": "set_figheight",
        "set_xlabel": "set_ylabel",
        "set_xlim": "set_ylim",
        "set_ylabel": "set_xlabel",
        "set_ylim": "set_xlim",
        "sharex": "sharey",
        "sharey": "sharex",
        "top": "right",
        "vlines": "hlines",
        "width": "height",
        "wspace": "hspace",
        "xaxis": "yaxis",
        "yaxis": "xaxis",
    }


def _transpose(obj):
    if isinstance(obj, str):
        return _Transposed._NAME_TRANSPOSE.get(obj, obj)
    return _Transposed(obj)


def _identity(obj):
    return obj


class UpSet:
    """Manage the data and drawing for a basic UpSet plot

    Primary public method is :meth:`plot`.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Elements associated with categories (a DataFrame), or the size of each
        subset of categories (a Series).
        Should have MultiIndex where each level is binary,
        corresponding to category membership.
        If a DataFrame, `sum_over` must be a string or False.
    orientation : {'horizontal' (default), 'vertical'}
        If horizontal, intersections are listed from left to right.
    sort_by : {'cardinality', 'degree', '-cardinality', '-degree',
               'input', '-input'}
        If 'cardinality', subset are listed from largest to smallest.
        If 'degree', they are listed in order of the number of categories
        intersected. If 'input', the order they appear in the data input is
        used.
        Prefix with '-' to reverse the ordering.

        Note this affects ``subset_sizes`` but not ``data``.
    sort_categories_by : {'cardinality', '-cardinality', 'input', '-input'}
        Whether to sort the categories by total cardinality, or leave them
        in the input data's provided order (order of index levels).
        Prefix with '-' to reverse the ordering.
    subset_size : {'auto', 'count', 'sum'}
        Configures how to calculate the size of a subset. Choices are:

        'auto' (default)
            If `data` is a DataFrame, count the number of rows in each group,
            unless `sum_over` is specified.
            If `data` is a Series with at most one row for each group, use
            the value of the Series. If `data` is a Series with more than one
            row per group, raise a ValueError.
        'count'
            Count the number of rows in each group.
        'sum'
            Sum the value of the `data` Series, or the DataFrame field
            specified by `sum_over`.
    sum_over : str or None
        If `subset_size='sum'` or `'auto'`, then the intersection size is the
        sum of the specified field in the `data` DataFrame. If a Series, only
        None is supported and its value is summed.
    min_subset_size : int or "number%", optional
        Minimum size of a subset to be shown in the plot. All subsets with
        a size smaller than this threshold will be omitted from plotting.
        This may be specified as a percentage
        using a string, like "50%".
        Size may be a sum of values, see `subset_size`.

        .. versionadded:: 0.5

        .. versionchanged:: 0.9
            Support percentages
    max_subset_size : int or "number%", optional
        Maximum size of a subset to be shown in the plot. All subsets with
        a size greater than this threshold will be omitted from plotting.
        This may be specified as a percentage
        using a string, like "50%".

        .. versionadded:: 0.5

        .. versionchanged:: 0.9
            Support percentages
    max_subset_rank : int, optional
        Limit to the top N ranked subsets in descending order of size.
        All tied subsets are included.

        .. versionadded:: 0.9
    min_degree : int, optional
        Minimum degree of a subset to be shown in the plot.

        .. versionadded:: 0.5
    max_degree : int, optional
        Maximum degree of a subset to be shown in the plot.

        .. versionadded:: 0.5
    facecolor : 'auto' or matplotlib color or float
        Color for bar charts and active dots. Defaults to black if
        axes.facecolor is a light color, otherwise white.

        .. versionchanged:: 0.6
            Before 0.6, the default was 'black'
    other_dots_color : matplotlib color or float
        Color for shading of inactive dots, or opacity (between 0 and 1)
        applied to facecolor.

        .. versionadded:: 0.6
    shading_color : matplotlib color or float
        Color for shading of odd rows in matrix and totals, or opacity (between
        0 and 1) applied to facecolor.

        .. versionadded:: 0.6
    with_lines : bool
        Whether to show lines joining dots in the matrix, to mark multiple
        categories being intersected.
    element_size : float or None
        Side length in pt. If None, size is estimated to fit figure
    intersection_plot_elements : int
        The intersections plot should be large enough to fit this many matrix
        elements. Set to 0 to disable intersection size bars.

        .. versionchanged:: 0.4
            Setting to 0 is handled.
    totals_plot_elements : int
        The totals plot should be large enough to fit this many matrix
        elements. Set to 0 to disable the totals plot.

        .. versionchanged:: 0.9
            Setting to 0 is handled.
    show_counts : bool or str, default=False
        Whether to label the intersection size bars with the cardinality
        of the intersection. When a string, this formats the number.
        For example, '{:d}' is equivalent to True.
        Note that, for legacy reasons, if the string does not contain '{',
        it will be interpreted as a C-style format string, such as '%d'.
    show_percentages : bool or str, default=False
        Whether to label the intersection size bars with the percentage
        of the intersection relative to the total dataset.
        When a string, this formats the number representing a fraction of
        samples.
        For example, '{:.1%}' is the default, formatting .123 as 12.3%.
        This may be applied with or without show_counts.

        .. versionadded:: 0.4
    include_empty_subsets : bool (default=False)
        If True, all possible category combinations will be shown as subsets,
        even when some are not present in data.
    """

    _default_figsize = (10, 6)
    DPI = 100  # standard matplotlib value

    def __init__(
        self,
        data,
        orientation="horizontal",
        sort_by="degree",
        sort_categories_by="cardinality",
        subset_size="auto",
        sum_over=None,
        min_subset_size=None,
        max_subset_size=None,
        max_subset_rank=None,
        min_degree=None,
        max_degree=None,
        facecolor="auto",
        other_dots_color=0.18,
        shading_color=0.05,
        with_lines=True,
        element_size=32,
        intersection_plot_elements=6,
        totals_plot_elements=2,
        show_counts="",
        show_percentages=False,
        include_empty_subsets=False,
    ):
        self._horizontal = orientation == "horizontal"
        self._reorient = _identity if self._horizontal else _transpose
        if facecolor == "auto":
            bgcolor = matplotlib.rcParams.get("axes.facecolor", "white")
            r, g, b, a = colors.to_rgba(bgcolor)
            lightness = colors.rgb_to_hsv((r, g, b))[-1] * a
            facecolor = "black" if lightness >= 0.5 else "white"
        self._facecolor = facecolor
        self._shading_color = (
            _multiply_alpha(facecolor, shading_color)
            if isinstance(shading_color, float)
            else shading_color
        )
        self._other_dots_color = (
            _multiply_alpha(facecolor, other_dots_color)
            if isinstance(other_dots_color, float)
            else other_dots_color
        )
        self._with_lines = with_lines
        self._element_size = element_size
        self._totals_plot_elements = totals_plot_elements
        self._subset_plots = [
            {
                "type": "default",
                "id": "intersections",
                "elements": intersection_plot_elements,
            }
        ]
        if not intersection_plot_elements:
            self._subset_plots.pop()
        self._show_counts = show_counts
        self._show_percentages = show_percentages

        (self.total, self._df, self.intersections, self.totals) = _process_data(
            data,
            sort_by=sort_by,
            sort_categories_by=sort_categories_by,
            subset_size=subset_size,
            sum_over=sum_over,
            min_subset_size=min_subset_size,
            max_subset_size=max_subset_size,
            max_subset_rank=max_subset_rank,
            min_degree=min_degree,
            max_degree=max_degree,
            reverse=not self._horizontal,
            include_empty_subsets=include_empty_subsets,
        )
        self.category_styles = {}
        self.subset_styles = [{"facecolor": facecolor} for i in range(len(self.intersections))]
        self.subset_legend = []  # pairs of (style, label)

    def _swapaxes(self, x, y):
        if self._horizontal:
            return x, y
        return y, x

    def style_subsets(
        self,
        present=None,
        absent=None,
        min_subset_size=None,
        max_subset_size=None,
        max_subset_rank=None,
        min_degree=None,
        max_degree=None,
        facecolor=None,
        edgecolor=None,
        hatch=None,
        linewidth=None,
        linestyle=None,
        label=None,
    ):
        """Updates the style of selected subsets' bars and matrix dots

        Parameters are either used to select subsets, or to style them with
        attributes of :class:`matplotlib.patches.Patch`, apart from label,
        which adds a legend entry.

        Parameters
        ----------
        present : str or list of str, optional
            Category or categories that must be present in subsets for styling.
        absent : str or list of str, optional
            Category or categories that must not be present in subsets for
            styling.
        min_subset_size : int or "number%", optional
            Minimum size of a subset to be styled.
            This may be specified as a percentage using a string, like "50%".

            .. versionchanged:: 0.9
                Support percentages
        max_subset_size : int or "number%", optional
            Maximum size of a subset to be styled.
            This may be specified as a percentage using a string, like "50%".

            .. versionchanged:: 0.9
                Support percentages
        max_subset_rank : int, optional
            Limit to the top N ranked subsets in descending order of size.
            All tied subsets are included.

            .. versionadded:: 0.9
        min_degree : int, optional
            Minimum degree of a subset to be styled.
        max_degree : int, optional
            Maximum degree of a subset to be styled.

        facecolor : str or matplotlib color, optional
            Override the default UpSet facecolor for selected subsets.
        edgecolor : str or matplotlib color, optional
            Set the edgecolor for bars, dots, and the line between dots.
        hatch : str, optional
            Set the hatch. This will apply to intersection size bars, but not
            to matrix dots.
        linewidth : int, optional
            Line width in points for edges.
        linestyle : str, optional
            Line style for edges.

        label : str, optional
            If provided, a legend will be added
        """
        style = {
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "hatch": hatch,
            "linewidth": linewidth,
            "linestyle": linestyle,
        }
        style = {k: v for k, v in style.items() if v is not None}
        mask = _get_subset_mask(
            self.intersections,
            present=present,
            absent=absent,
            min_subset_size=min_subset_size,
            max_subset_size=max_subset_size,
            max_subset_rank=max_subset_rank,
            min_degree=min_degree,
            max_degree=max_degree,
        )
        for idx in np.flatnonzero(mask):
            self.subset_styles[idx].update(style)

        if label is not None:
            if "facecolor" not in style:
                style["facecolor"] = self._facecolor
            for i, (other_style, other_label) in enumerate(self.subset_legend):
                if other_style == style:
                    if other_label != label:
                        self.subset_legend[i] = (style, other_label + "; " + label)
                    break
            else:
                self.subset_legend.append((style, label))

    def _plot_bars(self, ax, data, title, colors=None, use_labels=False):
        ax = self._reorient(ax)
        ax.set_autoscalex_on(False)
        data_df = pd.DataFrame(data)
        if self._horizontal:
            data_df = data_df.loc[:, ::-1]  # reverse: top row is top of stack

        # TODO: colors should be broadcastable to data_df shape
        if callable(colors):
            colors = colors(range(data_df.shape[1]))
        elif isinstance(colors, (str, type(None))):
            colors = [colors] * len(data_df)

        if self._horizontal:
            colors = reversed(colors)

        x = np.arange(len(data_df))
        cum_y = None
        all_rects = []
        for (name, y), color in zip(data_df.items(), colors):
            rects = ax.bar(
                x,
                y,
                0.5,
                cum_y,
                color=color,
                zorder=10,
                label=name if use_labels else None,
                align="center",
            )
            cum_y = y if cum_y is None else cum_y + y
            all_rects.extend(rects)

        self._label_sizes(ax, rects, "top" if self._horizontal else "right")

        ax.xaxis.set_visible(False)
        for x in ["top", "bottom", "right"]:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)
        ax.set_ylabel(title)
        return all_rects

    def _plot_stacked_bars(self, ax, by, sum_over, colors, title):
        df = self._df.set_index("_bin").set_index(by, append=True, drop=False)
        gb = df.groupby(level=list(range(df.index.nlevels)), sort=True)
        if sum_over is None and "_value" in df.columns:
            data = gb["_value"].sum()
        elif sum_over is None:
            data = gb.size()
        else:
            data = gb[sum_over].sum()
        data = data.unstack(by).fillna(0)
        if isinstance(colors, str):
            colors = matplotlib.cm.get_cmap(colors)
        elif isinstance(colors, typing.Mapping):
            colors = data.columns.map(colors).values
            if pd.isna(colors).any():
                raise KeyError(
                    "Some labels mapped by colors: %r" % data.columns[pd.isna(colors)].tolist()
                )

        self._plot_bars(ax, data=data, colors=colors, title=title, use_labels=True)

        handles, labels = ax.get_legend_handles_labels()
        if self._horizontal:
            # Make legend order match visual stack order
            ax.legend(list(reversed(handles)), list(reversed(labels)))
        else:
            ax.legend()

    def add_stacked_bars(self, by, sum_over=None, colors=None, elements=3, title=None):
        """Add a stacked bar chart over subsets when :func:`plot` is called.

        Used to plot categorical variable distributions within each subset.

        .. versionadded:: 0.6

        Parameters
        ----------
        by : str
            Column name within the dataframe for color coding the stacked bars,
            containing discrete or categorical values.
        sum_over : str, optional
            Ordinarily the bars will chart the size of each group. sum_over
            may specify a column which will be summed to determine the size
            of each bar.
        colors : Mapping, list-like, str or callable, optional
            The facecolors to use for bars corresponding to each discrete
            label, specified as one of:

            Mapping
                Maps from label to matplotlib-compatible color specification.
            list-like
                A list of matplotlib colors to apply to labels in order.
            str
                The name of a matplotlib colormap name.
            callable
                When called with the number of labels, this should return a
                list-like of that many colors.  Matplotlib colormaps satisfy
                this callable API.
            None
                Uses the matplotlib default colormap.
        elements : int, default=3
            Size of the axes counted in number of matrix elements.
        title : str, optional
            The axis title labelling bar length.

        Returns
        -------
        None
        """
        # TODO: allow sort_by = {"lexical", "sum_squares", "rev_sum_squares",
        #                        list of labels}
        self._subset_plots.append(
            {
                "type": "stacked_bars",
                "by": by,
                "sum_over": sum_over,
                "colors": colors,
                "title": title,
                "id": "extra%d" % len(self._subset_plots),
                "elements": elements,
            }
        )

    def add_catplot(self, kind, value=None, elements=3, **kw):
        """Add a seaborn catplot over subsets when :func:`plot` is called.

        Parameters
        ----------
        kind : str
            One of {"point", "bar", "strip", "swarm", "box", "violin", "boxen"}
        value : str, optional
            Column name for the value to plot (i.e. y if
            orientation='horizontal'), required if `data` is a DataFrame.
        elements : int, default=3
            Size of the axes counted in number of matrix elements.
        **kw : dict
            Additional keywords to pass to :func:`seaborn.catplot`.

            Our implementation automatically determines 'ax', 'data', 'x', 'y'
            and 'orient', so these are prohibited keys in `kw`.

        Returns
        -------
        None
        """
        assert not set(kw.keys()) & {"ax", "data", "x", "y", "orient"}
        if value is None:
            if "_value" not in self._df.columns:
                raise ValueError("value cannot be set if data is a Series. Got %r" % value)
        else:
            if value not in self._df.columns:
                raise ValueError("value %r is not a column in data" % value)
        self._subset_plots.append(
            {
                "type": "catplot",
                "value": value,
                "kind": kind,
                "id": "extra%d" % len(self._subset_plots),
                "elements": elements,
                "kw": kw,
            }
        )

    def _check_value(self, value):
        if value is None and "_value" in self._df.columns:
            value = "_value"
        elif value is None:
            raise ValueError("value can only be None when data is a Series")
        return value

    def _plot_catplot(self, ax, value, kind, kw):
        df = self._df
        value = self._check_value(value)
        kw = kw.copy()
        if self._horizontal:
            kw["orient"] = "v"
            kw["x"] = "_bin"
            kw["y"] = value
        else:
            kw["orient"] = "h"
            kw["x"] = value
            kw["y"] = "_bin"
        import seaborn

        kw["ax"] = ax
        getattr(seaborn, kind + "plot")(data=df, **kw)

        ax = self._reorient(ax)
        if value == "_value":
            ax.set_ylabel("")

        ax.xaxis.set_visible(False)
        for x in ["top", "bottom", "right"]:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)

    def make_grid(self, fig=None):
        """Get a SubplotSpec for each Axes, accounting for label text width"""
        n_cats = len(self.totals)
        n_inters = len(self.intersections)

        if fig is None:
            fig = plt.gcf()

        # Determine text size to determine figure size / spacing
        text_kw = {"size": matplotlib.rcParams["xtick.labelsize"]}
        # adding "x" ensures a margin
        t = fig.text(
            0,
            0,
            "\n".join(str(label) + "x" for label in self.totals.index.values),
            **text_kw,
        )
        window_extent_args = {}
        if RENDERER_IMPORTED:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                window_extent_args["renderer"] = get_renderer(fig)
        textw = t.get_window_extent(**window_extent_args).width
        t.remove()

        window_extent_args = {}
        if RENDERER_IMPORTED:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                window_extent_args["renderer"] = get_renderer(fig)
        figw = self._reorient(fig.get_window_extent(**window_extent_args)).width

        sizes = np.asarray([p["elements"] for p in self._subset_plots])
        fig = self._reorient(fig)

        non_text_nelems = len(self.intersections) + self._totals_plot_elements
        if self._element_size is None:
            colw = (figw - textw) / non_text_nelems
        else:
            render_ratio = figw / fig.get_figwidth()
            colw = self._element_size / 72 * render_ratio
            figw = colw * (non_text_nelems + np.ceil(textw / colw) + 1)
            fig.set_figwidth(figw / render_ratio)
            fig.set_figheight((colw * (n_cats + sizes.sum())) / render_ratio)

        text_nelems = int(np.ceil(figw / colw - non_text_nelems))

        GS = self._reorient(matplotlib.gridspec.GridSpec)
        gridspec = GS(
            *self._swapaxes(
                n_cats + (sizes.sum() or 0),
                n_inters + text_nelems + self._totals_plot_elements,
            ),
            hspace=1,
        )
        if self._horizontal:
            out = {
                "matrix": gridspec[-n_cats:, -n_inters:],
                "shading": gridspec[-n_cats:, :],
                "totals": None
                if self._totals_plot_elements == 0
                else gridspec[-n_cats:, : self._totals_plot_elements],
                "gs": gridspec,
            }
            cumsizes = np.cumsum(sizes[::-1])
            for start, stop, plot in zip(
                np.hstack([[0], cumsizes]), cumsizes, self._subset_plots[::-1]
            ):
                out[plot["id"]] = gridspec[start:stop, -n_inters:]
        else:
            out = {
                "matrix": gridspec[-n_inters:, :n_cats],
                "shading": gridspec[:, :n_cats],
                "totals": None
                if self._totals_plot_elements == 0
                else gridspec[: self._totals_plot_elements, :n_cats],
                "gs": gridspec,
            }
            cumsizes = np.cumsum(sizes)
            for start, stop, plot in zip(np.hstack([[0], cumsizes]), cumsizes, self._subset_plots):
                out[plot["id"]] = gridspec[-n_inters:, start + n_cats : stop + n_cats]
        return out

    def plot_matrix(self, ax):
        """Plot the matrix of intersection indicators onto ax"""
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels

        inclusion = data.index.to_frame().values

        # Prepare styling
        styles = [
            [
                self.subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": self._other_dots_color, "linewidth": 0}
                for j in range(n_cats)
            ]
            for i in range(len(data))
        ]
        styles = sum(styles, [])  # flatten nested list
        style_columns = {
            "facecolor": "facecolors",
            "edgecolor": "edgecolors",
            "linewidth": "linewidths",
            "linestyle": "linestyles",
            "hatch": "hatch",
        }
        styles = (
            pd.DataFrame(styles)
            .reindex(columns=style_columns.keys())
            .astype(
                {
                    "facecolor": "O",
                    "edgecolor": "O",
                    "linewidth": float,
                    "linestyle": "O",
                    "hatch": "O",
                }
            )
        )
        styles["linewidth"].fillna(1, inplace=True)
        styles["facecolor"].fillna(self._facecolor, inplace=True)
        styles["edgecolor"].fillna(styles["facecolor"], inplace=True)
        styles["linestyle"].fillna("solid", inplace=True)
        del styles["hatch"]  # not supported in matrix (currently)

        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))

        # Plot dots
        if self._element_size is not None:  # noqa
            s = (self._element_size * 0.35) ** 2
        else:
            # TODO: make s relative to colw
            s = 200
        ax.scatter(
            *self._swapaxes(x, y),
            s=s,
            zorder=10,
            **styles.rename(columns=style_columns),
        )

        # Plot lines
        if self._with_lines:
            idx = np.flatnonzero(inclusion)
            line_data = pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
            colors = pd.Series(
                [
                    style.get("edgecolor", style.get("facecolor", self._facecolor))
                    for style in self.subset_styles
                ],
                name="color",
            )
            line_data = line_data.join(colors)
            ax.vlines(
                line_data.index.values,
                line_data["min"],
                line_data["max"],
                lw=2,
                colors=line_data["color"],
                zorder=5,
            )

        # Ticks and axes
        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_axis.set_ticklabels(data.index.names, rotation=0 if self._horizontal else -90)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position("top")
        ax.set_frame_on(False)
        ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
        ax.grid(False)

    def plot_intersections(self, ax):
        """Plot bars indicating intersection size"""
        rects = self._plot_bars(
            ax, self.intersections, title="Intersection size", colors=self._facecolor
        )
        for style, rect in zip(self.subset_styles, rects):
            style = style.copy()
            style.setdefault("edgecolor", style.get("facecolor", self._facecolor))
            for attr, val in style.items():
                getattr(rect, "set_" + attr)(val)

        if self.subset_legend:
            styles, labels = zip(*self.subset_legend)
            styles = [patches.Patch(**patch_style) for patch_style in styles]
            ax.legend(styles, labels)

    def _label_sizes(self, ax, rects, where):
        if not self._show_counts and not self._show_percentages:
            return
        if self._show_counts is True:
            count_fmt = "{:.0f}"
        else:
            count_fmt = self._show_counts
            if "{" not in count_fmt:
                count_fmt = util.to_new_pos_format(count_fmt)

        pct_fmt = "{:.1%}" if self._show_percentages is True else self._show_percentages

        if count_fmt and pct_fmt:
            if where == "top":
                fmt = f"{count_fmt}\n({pct_fmt})"
            else:
                fmt = f"{count_fmt} ({pct_fmt})"

            def make_args(val):
                return val, val / self.total
        elif count_fmt:
            fmt = count_fmt

            def make_args(val):
                return (val,)
        else:
            fmt = pct_fmt

            def make_args(val):
                return (val / self.total,)

        if where == "right":
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width() + rect.get_x()
                ax.text(
                    width + margin,
                    rect.get_y() + rect.get_height() * 0.5,
                    fmt.format(*make_args(width)),
                    ha="left",
                    va="center",
                )
        elif where == "left":
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width() + rect.get_x()
                ax.text(
                    width + margin,
                    rect.get_y() + rect.get_height() * 0.5,
                    fmt.format(*make_args(width)),
                    ha="right",
                    va="center",
                )
        elif where == "top":
            margin = 0.01 * abs(np.diff(ax.get_ylim()))
            for rect in rects:
                height = rect.get_height() + rect.get_y()
                ax.text(
                    rect.get_x() + rect.get_width() * 0.5,
                    height + margin,
                    fmt.format(*make_args(height)),
                    ha="center",
                    va="bottom",
                )
        else:
            raise NotImplementedError("unhandled where: %r" % where)

    def plot_totals(self, ax):
        """Plot bars indicating total set size"""
        orig_ax = ax
        ax = self._reorient(ax)
        rects = ax.barh(
            np.arange(len(self.totals.index.values)),
            self.totals,
            0.5,
            color=self._facecolor,
            align="center",
        )
        self._label_sizes(ax, rects, "left" if self._horizontal else "top")

        for category, rect in zip(self.totals.index.values, rects):
            style = {
                k[len("bar_") :]: v
                for k, v in self.category_styles.get(category, {}).items()
                if k.startswith("bar_")
            }
            style.setdefault("edgecolor", style.get("facecolor", self._facecolor))
            for attr, val in style.items():
                getattr(rect, "set_" + attr)(val)

        max_total = self.totals.max()
        if self._horizontal:
            orig_ax.set_xlim(max_total, 0)
        for x in ["top", "left", "right"]:
            ax.spines[self._reorient(x)].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.patch.set_visible(False)

    def plot_shading(self, ax):
        # shade all rows, set every second row to zero visibility
        for i, category in enumerate(self.totals.index):
            default_shading = self._shading_color if i % 2 == 0 else (0.0, 0.0, 0.0, 0.0)
            shading_style = {
                k[len("shading_") :]: v
                for k, v in self.category_styles.get(category, {}).items()
                if k.startswith("shading_")
            }

            lw = shading_style.get("linewidth", 1 if shading_style.get("edgecolor") else 0)
            lw_padding = lw / (self._default_figsize[0] * self.DPI)
            start_x = lw_padding
            end_x = 1 - lw_padding * 3

            rect = plt.Rectangle(
                self._swapaxes(start_x, i - 0.4),
                *self._swapaxes(end_x, 0.8),
                facecolor=shading_style.get("facecolor", default_shading),
                edgecolor=shading_style.get("edgecolor", None),
                ls=shading_style.get("linestyle", "-"),
                lw=lw,
                zorder=0,
            )

            ax.add_patch(rect)
        ax.set_frame_on(False)
        ax.tick_params(
            axis="both",
            which="both",
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def style_categories(
        self,
        categories,
        *,
        bar_facecolor=None,
        bar_hatch=None,
        bar_edgecolor=None,
        bar_linewidth=None,
        bar_linestyle=None,
        shading_facecolor=None,
        shading_edgecolor=None,
        shading_linewidth=None,
        shading_linestyle=None,
    ):
        """Updates the style of the categories.

        Select a category by name, and style either its total bar or its shading.

        .. versionadded:: 0.9

        Parameters
        ----------
        categories : str or list[str]
            Category names where the changed style applies.
        bar_facecolor : str or RGBA matplotlib color tuple, optional.
            Override the default facecolor in the totals plot.
        bar_hatch : str, optional
            Set a hatch for the totals plot.
        bar_edgecolor : str or matplotlib color, optional
            Set the edgecolor for total bars.
        bar_linewidth : int, optional
            Line width in points for total bar edges.
        bar_linestyle : str, optional
            Line style for edges.
        shading_facecolor : str or RGBA matplotlib color tuple, optional.
            Override the default alternating shading for specified categories.
        shading_edgecolor : str or matplotlib color, optional
            Set the edgecolor for bars, dots, and the line between dots.
        shading_linewidth : int, optional
            Line width in points for edges.
        shading_linestyle : str, optional
            Line style for edges.
        """
        if isinstance(categories, str):
            categories = [categories]
        style = {
            "bar_facecolor": bar_facecolor,
            "bar_hatch": bar_hatch,
            "bar_edgecolor": bar_edgecolor,
            "bar_linewidth": bar_linewidth,
            "bar_linestyle": bar_linestyle,
            "shading_facecolor": shading_facecolor,
            "shading_edgecolor": shading_edgecolor,
            "shading_linewidth": shading_linewidth,
            "shading_linestyle": shading_linestyle,
        }
        style = {k: v for k, v in style.items() if v is not None}
        for category_name in categories:
            self.category_styles.setdefault(category_name, {}).update(style)

    def plot(self, fig=None):
        """Draw all parts of the plot onto fig or a new figure

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Defaults to a new figure.

        Returns
        -------
        subplots : dict of matplotlib.axes.Axes
            Keys are 'matrix', 'intersections', 'totals', 'shading'
        """
        if fig is None:
            fig = plt.figure(figsize=self._default_figsize)
        specs = self.make_grid(fig)
        shading_ax = fig.add_subplot(specs["shading"])
        self.plot_shading(shading_ax)
        matrix_ax = self._reorient(fig.add_subplot)(specs["matrix"], sharey=shading_ax)
        self.plot_matrix(matrix_ax)
        if specs["totals"] is None:
            totals_ax = None
        else:
            totals_ax = self._reorient(fig.add_subplot)(specs["totals"], sharey=matrix_ax)
            self.plot_totals(totals_ax)
        out = {"matrix": matrix_ax, "shading": shading_ax, "totals": totals_ax}

        for plot in self._subset_plots:
            ax = self._reorient(fig.add_subplot)(specs[plot["id"]], sharex=matrix_ax)
            if plot["type"] == "default":
                self.plot_intersections(ax)
            elif plot["type"] in self.PLOT_TYPES:
                kw = plot.copy()
                del kw["type"]
                del kw["elements"]
                del kw["id"]
                self.PLOT_TYPES[plot["type"]](self, ax, **kw)
            else:
                raise ValueError("Unknown subset plot type: %r" % plot["type"])
            out[plot["id"]] = ax

        self._reorient(fig).align_ylabels([out[plot["id"]] for plot in self._subset_plots])
        return out

    PLOT_TYPES = {
        "catplot": _plot_catplot,
        "stacked_bars": _plot_stacked_bars,
    }

    def _repr_html_(self):
        fig = plt.figure(figsize=self._default_figsize)
        self.plot(fig=fig)
        return fig._repr_html_()


def plot(data, fig=None, **kwargs):
    """Make an UpSet plot of data on fig

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
        If a DataFrame, `sum_over` must be a string or False.
    fig : matplotlib.figure.Figure, optional
        Defaults to a new figure.
    kwargs
        Other arguments for :class:`UpSet`

    Returns
    -------
    subplots : dict of matplotlib.axes.Axes
        Keys are 'matrix', 'intersections', 'totals', 'shading'
    """
    return UpSet(data, **kwargs).plot(fig)


#####################################################################################


def preprocess_materials_tools(df_all, occupations_file_names):
    tools_data = pd.DataFrame()
    materials_data = pd.DataFrame()
    for occ_group in occupations_file_names:
        materials, tools = get_shares(df_all[df_all["occupation_group"] == occ_group])
        materials.index = materials["Category"].str.replace("materials.", "", regex=True)
        materials.drop("Category", axis=1, inplace=True)

        tools.index = tools["Category"].str.replace("tools.", "", regex=True)
        tools.drop("Category", axis=1, inplace=True)
        materials_data = pd.concat(
            [materials_data, materials.rename(columns={"Value": occ_group})], axis=1
        )
        tools_data = pd.concat([tools_data, tools.rename(columns={"Value": occ_group})], axis=1)

    return materials_data, tools_data


def read_in_data(occupations_file_names, prefix, suffix):
    df_all = pd.DataFrame()
    for occ in occupations_file_names:
        df = pd.read_csv(prefix + occ + suffix)
        df["occupation_group"] = occ
        df_all = pd.concat([df_all, df], ignore_index=True)
    print(df_all.shape[0], " tasks in the data")
    return df_all


def plot_grouped_horizontal_bars(ax, data, title, colors, bar_height, occupations_file_names):
    categories = data["Category"].str.replace("materials.|tools.", "", regex=True).unique()
    groups = occupations_file_names

    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 18
    legend_fontsize = 18
    annot_fontsize = 14

    y = np.arange(len(categories))
    group_width = bar_height * len(groups)
    offsets = np.linspace(
        -group_width / 2 + bar_height / 2, group_width / 2 - bar_height / 2, len(groups)
    )

    ax.set_title(title, fontsize=title_fontsize)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Percentage", fontsize=label_fontsize)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=tick_fontsize)
    ax.tick_params(axis="x", which="major", labelsize=tick_fontsize)

    for spine in ax.spines.values():
        spine.set_visible(False)

    bars_list = []  # for legend handles

    for i, group in enumerate(groups):
        values = data[group].values

        bars = ax.barh(
            y + offsets[i],
            values,
            height=bar_height,
            color=colors[i],
            label=group.title().replace("_", " "),
            alpha=0.7,
            edgecolor=colors[i],
        )

        bars_list.append(bars[0])  # Save one bar per group for the legend

        for bar in bars:
            width = bar.get_width()
            label = f"{width * 100:.0f}%"
            if width > 0.2:
                ax.text(
                    width - 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="right",
                    va="center",
                    color="black",
                    fontsize=annot_fontsize,
                )
            else:
                ax.text(
                    width + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="left",
                    va="center",
                    fontsize=annot_fontsize,
                )

    return bars_list  # Return bar objects for legend


def get_shares(df):
    material_columns = df.columns[df.columns.str.startswith("materials")]
    material_columns = [c for c in material_columns if "Other" not in c]

    tool_columns = df.columns[df.columns.str.startswith("tools")]
    tool_columns = [c for c in tool_columns if "Other" not in c]

    relevant_columns = material_columns + tool_columns
    df_binary = (df[relevant_columns] == "Required").astype(int)

    df_shares = df_binary.sum() / df.shape[0]
    df_shares = (
        pd.DataFrame(df_shares).reset_index().rename(columns={0: "Value", "index": "Category"})
    )
    # df_shares.loc[
    #     df_shares["Category"] == "materials.Virtual labs or sandbox environments", "Category"
    # ] = "materials.Virtual labs"
    # Separate materials and tools
    materials = df_shares[df_shares["Category"].str.startswith("materials")]
    tools = df_shares[df_shares["Category"].str.startswith("tools")]

    return [materials, tools]


def plot_tools_materials_requirements(
    materials_data, tools_data, fig_path, occupations_file_names
):
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.4)  # Increase space between the two plots

    # Plot materials
    bars_materials = plot_grouped_horizontal_bars(
        axes[0],
        materials_data.reset_index(),
        "Materials",
        ["#FFBD59", "#38B6FF", "#8E3B46"],
        0.3,
        occupations_file_names,
    )

    # Plot tools
    bars_tools = plot_grouped_horizontal_bars(
        axes[1],
        tools_data.reset_index(),
        "Tools",
        ["#FFBD59", "#38B6FF", "#8E3B46"],
        0.3,
        occupations_file_names,
    )

    # Joint legend at top center
    fig.legend(
        bars_materials,  # or bars_tools (they're the same bar styles)
        labels=[group.title().replace("_", " ") for group in occupations_file_names],
        loc="upper center",
        ncol=len(occupations_file_names),
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),  # Center it above the plots
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Successfully saved figure to ", fig_path)


def mark_invalid(df):
    # mark exams with empty entry, nan entry or key grade scores over 100 as invalid

    df.loc[df["exam"] == "", "exam"] = "Exam not valid"
    df["exam_na"] = False
    df.loc[df["exam"].isna(), "exam_na"] = True
    df["exam"] = df["exam"].fillna("Exam not valid")
    df.loc[df["key_grade"] > 100, "exam"] = "Exam not valid"
    # df.loc[df["score_empty_submission"] > 0, "exam"] = "Exam not valid"
    df.loc[df["check_overall_makes_sense"] == False, "exam"] = "Exam not valid"
    df.loc[df["check_no_internet"] == False, "exam"] = "Exam not valid"
    df.loc[df["check_real_materials"] == False, "exam"] = "Exam not valid"
    score_columns = df.filter(like="score", axis=1).columns
    df.loc[np.any(df[score_columns] > 100, axis=1), "exam"] = "Exam not valid"
    return df


def create_exams_table(task_list, exams, table_path):
    # count number of tasks per occupation group
    overall_count = (
        task_list.groupby("occupation_group")["task_id"].nunique().rename("\# of tasks")
    )
    # count number of examined tasks per occupation group
    examined_tasks_count = (
        exams.groupby("occupation_group")["task_id"].nunique().rename("\# of examined tasks")
    )
    valid_exams = (
        exams[exams["exam"] != "Exam not valid"]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\# of valid exams")
    )
    print(exams[exams["exam"] != "Exam not valid"])
    # create table with overview statistics
    stats_table = pd.concat([overall_count, examined_tasks_count], axis=1)
    stats_table["\% of examined tasks"] = np.round(
        stats_table["\# of examined tasks"] / stats_table["\# of tasks"] * 100, 2
    )
    stats_table = pd.concat([stats_table, valid_exams], axis=1)
    stats_table["\% of valid exams"] = np.round(
        stats_table["\# of valid exams"] / stats_table["\# of examined tasks"] * 100, 2
    )
    stats_table.index = [group.title().replace("_", " ") for group in occupations_file_names]
    stats_table = stats_table.round(2)
    latex_code = stats_table.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=True,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully saved table to ", table_path)


def plot_exams_share(task_list, exams, fig_path):
    task_count = task_list.groupby("occupation").size()
    attempt_count = exams.groupby("occupation").size()
    success_count = exams[exams["exam"] != "Exam not valid"].groupby("occupation").size()

    counts = pd.concat([task_count, attempt_count, success_count], axis=1).rename(
        columns={0: "\# of tasks", 1: "\# of attempts", 2: "\# of successes"}
    )
    counts = counts.merge(
        df_all[["occupation", "occupation_group"]].drop_duplicates(), on="occupation", how="left"
    )
    counts = counts.sort_values(["occupation_group", "occupation"]).reset_index(drop=True)

    # Create a new DataFrame to store the final result
    new_rows = []
    i = 0
    blank_row_names = [" ", "  ", "   ", "    ", "     ", "      "]
    # Iterate over each group
    for group_name, group_df in counts.groupby("occupation_group"):
        # Create the label row
        label_row = {
            "occupation": group_name.replace("_", " ").title(),  # optional: prettify
            "\# of tasks": np.nan,
            "\# of attempts": np.nan,
            "\# of successes": np.nan,
            "occupation_group": group_name,
        }
        blank_row = {
            "occupation": blank_row_names[i],  # optional: prettify
            "\# of tasks": np.nan,
            "\# of attempts": np.nan,
            "\# of successes": np.nan,
            "occupation_group": np.nan,
        }
        new_rows.append(pd.DataFrame([blank_row]))
        i += 1
        # Append the label row and the group rows
        new_rows.append(pd.DataFrame([label_row]))
        blank_row = {
            "occupation": blank_row_names[i],  # optional: prettify
            "\# of tasks": np.nan,
            "\# of attempts": np.nan,
            "\# of successes": np.nan,
            "occupation_group": np.nan,
        }
        new_rows.append(pd.DataFrame([blank_row]))
        new_rows.append(group_df)
        i += 1

    # Concatenate all parts into one final DataFrame
    df_with_labels = pd.concat(new_rows, ignore_index=True)

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 8),
        # gridspec_kw={"height_ratios": [5, 3]}  # Top row is 2x the height of the bottom row
    )
    axes = axes.flatten()
    # bar_colors=[#FFBD59, '#38B6FF',#8E3B46]
    bar_colors = [
        ("#739E82", "\# of tasks", "# of core tasks"),
        ("#E0777D", "\# of attempts", "# of exams attempted"),
        ("#8E3B46", "\# of successes", "# of created exams"),
    ]

    # Plot for each occupation group
    for idx, group in enumerate(occupations_file_names):
        ax = axes[idx]
        df_occ = counts[counts["occupation_group"] == group]

        for color, name, label in bar_colors:
            sns.barplot(
                x=name,  # gets 'tasks', 'exams', 'successes'
                y="occupation",
                data=df_occ,
                label=label,
                color=color,
                alpha=0.8,
                edgecolor=color,
                ax=ax,
                legend=False,
            )

        ax.set(xlim=(0, 24), ylabel="", xlabel="# of core tasks per occupation")
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_title(group.replace("_", " ").title(), fontsize=14)

        # Clean up ticks
        ax.tick_params(left=False)
        ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()])

    # Fourth plot (bottom right) is used for legend only
    # legend_ax = axes[3]
    # legend_ax.axis('off')  # Hide the axis

    # Create dummy bars for the legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8) for color, name, label in bar_colors
    ]
    labels = [label for _, _, label in bar_colors]
    # legend_ax.legend(handles, labels, loc='upper left', fontsize=14, title="Legend")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=12,
        title="Legend",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        title_fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(pad=5.0)  # Increase padding between subplots
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def create_failure_modes_table(df, table_path):
    task_count = df.groupby("occupation_group")["task_id"].nunique().rename("tasks")
    invalid_exams = (
        df[df["exam"] == "Exam not valid"]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\# of invalid exams")
    )
    key_grade_low = (
        df[df["key_grade"] < 99]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of answer keys with score \< 99")
    )
    key_grade_high = (
        df[df["key_grade"] > 100]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of answer keys with score \> 100")
    )
    check_no_internet = (
        df[df["check_no_internet"] == False]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of exams relying on fabricated websites")
    )
    failed_candidate_materials = (
        df[df["failed_candidate_materials"] > 2]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of exam without separate candidate materials")
    )
    check_overall_makes_sense = (
        df[df["check_overall_makes_sense"] == False]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of exams not making sense overall")
    )
    check_real_materials = (
        df[df["check_real_materials"] == False]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of exams relying on non-existing image materials")
    )

    exam_na = (
        df[df["exam_na"] == True]
        .groupby("occupation_group")["task_id"]
        .nunique()
        .rename("\% of exam generations failing during the process")
    )
    failure_modes = pd.concat(
        [
            invalid_exams,
            key_grade_low,
            key_grade_high,
            check_no_internet,
            failed_candidate_materials,
            check_overall_makes_sense,
            check_real_materials,
            exam_na,
            task_count,
        ],
        axis=1,
    ).fillna(0)
    failure_modes = failure_modes.div(failure_modes["tasks"], axis=0).drop(
        ["\# of invalid exams", "tasks"], axis=1
    )
    failure_modes = np.round(failure_modes * 100, 2)
    failure_modes.index = [group.title().replace("_", " ") for group in occupations_file_names]
    latex_code = failure_modes.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=True,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully saved table to ", table_path)


def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def contains_target_values(entry):
    llm_tools = []
    try:
        parsed_list = ast.literal_eval(entry)  # safely parse string to list
        if any(item in parsed_list for item in ["Online search engine"]):
            llm_tools.append("Web search")
        if any(item in parsed_list for item in ["Code execution environment"]):
            llm_tools.append("Bash and Python")
        if any(item in parsed_list for item in ["Image viewer", "Email client"]):
            llm_tools.append("Function calling")
        if any(
            item in parsed_list
            for item in [
                "Presentation software",
                "Video conferencing",
                "Chat interfaces",
                "Specialized software",
            ]
        ):
            llm_tools.append("Computer use")
        if any(item in parsed_list for item in ["Image generator"]):
            llm_tools.append("Image generation")
        if any(item in parsed_list for item in ["Internet access"]):
            llm_tools.append("Web browser")
        return llm_tools
    except (ValueError, SyntaxError):
        return False  # handle malformed strings


def ensure_iterable(x):
    if isinstance(x, bool):  # or you can check if not iterable
        return []
    elif isinstance(x, str):
        return [x]  # wrap string into list
    else:
        return x


def plot_upset(df, col, title, fig_path):
    # create mapping to LLM tools
    df["LLM_tools"] = df["required_tools"].apply(contains_target_values)
    df["LLM_tools"] = df["LLM_tools"].apply(ensure_iterable)

    if col == "LLM_tools":
        index_list = [
            "Bash and Python",
            "Function calling",
            "Computer use",
            "Image generation",
            "Web search",
            "Web browser",
        ]
    else:
        index_list = [
            "Text",
            "Data",
            "Images",
            "Audio files",
            "Video files",
            "Virtual labs",
            "Specialized files",
        ]
    # Convert list of label lists to a binary indicator DataFrame
    mlb = MultiLabelBinarizer()
    df.dropna(subset=col, inplace=True)
    binary_matrix = mlb.fit_transform(df[col].apply(safe_eval))

    # mlb.fit_transform(list(experiments['required_materials'].apply(ast.literal_eval)))

    df_labels = pd.DataFrame(binary_matrix, columns=mlb.classes_).astype(bool)
    df_labels = pd.concat([df_labels, df["occupation_group"].reset_index()], axis=1).drop(
        "index", axis=1
    )

    multi_index = pd.MultiIndex.from_frame(df_labels[index_list], names=index_list)
    data = df_labels.set_index(multi_index)
    data["occupation_group"] = data["occupation_group"].map(
        {
            "business_and_financial_operations_occupations": "Business and Financial Operations",
            "computer_and_mathematical_occupations": "Computer and Mathematical",
            "management_occupations": "Management",
        }
    )

    bar_colors = ["#8E3B46", "#38B6FF", "#FFBD59"]

    # Your long labels with manual line breaks for two lines
    labels = ["Business and\nFinancial Operations", "Computer and\nMathematical", "Management"]

    patches = [Patch(color=bar_colors[i], label=labels[i]) for i in range(len(labels))]

    # Create and plot the UpSet plot
    upset = UpSet(
        data,
        sort_by="cardinality",
        intersection_plot_elements=0,
        show_counts=True,
        min_subset_size=10,
    )
    upset.add_stacked_bars(
        by="occupation_group",
        title="Count by occupation group",
        elements=10,
        colors=bar_colors,
    )
    # upset.style_categories(
    #     "Practical exam possible",
    #     shading_facecolor="lavender",
    # )
    # if col == "LLM_tools":
    #     upset.style_subsets(
    #         absent=[
    #             "Bash and Python",
    #             "Computer use",
    #             "Image generation",
    #             "Web search",
    #             "Web browser",
    #         ],
    #         # facecolor="#739E82",
    #         #edgecolor="red",
    #         hatch="/",
    #     )
    if col == "required_materials":
        upset.style_subsets(
            present=["Text", "Data"],
            absent=["Images", "Audio files", "Video files", "Virtual labs", "Specialized files"],
            facecolor="#739E82",
            # edgecolor="red",
            # hatch="/",
        )

    upset.plot()

    # Add the legend once with custom patches and position
    if col == "LLM_tools":
        # plt.legend(
        #     handles=patches, title="Occupation Group", bbox_to_anchor=(-0.5, 1), loc="upper left"
        # )
        plt.gca().legend_.remove()

    # if col == "required_materials":

    plt.title(title)

    # Adjust layout
    plt.tight_layout(pad=1.0)  # Increase padding between subplots
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        # model = "claude-3-7-sonnet-20250219"
        # model = "o3-2025-04-16"
        model = "gemini-2.5-pro"

    occupations = [
        "Business and Financial Operations Occupations",
        "Computer and Mathematical Occupations",
        "Management Occupations",
    ]
    occupations_file_names = [occ.lower().replace(" ", "_") for occ in occupations]

    # df_all = read_in_data(
    #     occupations_file_names,
    #     f"../data/required_materials_tools/{model}/materials_tools_",
    #     "_CORE.csv",
    # )

    # # preprocess data to extract materials and tools lists
    # materials_data, tools_data = preprocess_materials_tools(df_all, occupations_file_names)

    # # create figure showing percetages of tools and materials required for each occupation group
    # plot_tools_materials_requirements(
    #     materials_data,
    #     tools_data,
    #     f"../results/figures/{model}/{model}_tools_materials_requirements.pdf",
    #     occupations_file_names,
    # )
    df_all = read_in_data(
        occupations_file_names,
        f"../data/exams/basic/{model}/",
        "_exams.csv",
    )
    # # get exams
    exam_list = read_in_data(occupations_file_names, f"../data/exams/basic/{model}/", "_exams.csv")
    exam_list = mark_invalid(exam_list)
    create_exams_table(df_all, exam_list, f"../results/tables/{model}/{model}_exams_overview.tex")
    plot_exams_share(
        df_all, exam_list, f"../results/figures/{model}/{model}_exams_shares_overview.pdf"
    )
    create_failure_modes_table(
        exam_list, f"../results/tables/{model}/{model}_exams_failure_modes.tex"
    )

    plot_upset(
        df_all,
        "LLM_tools",
        "Tool co-occurrence by occupation group",
        f"../results/figures/{model}/{model}_upset_LLM_tools.pdf",
    )
    plot_upset(
        df_all,
        "required_materials",
        "Material co-occurrence by occupation group",
        f"../results/figures/{model}/{model}_upset_materials_required.pdf",
    )
