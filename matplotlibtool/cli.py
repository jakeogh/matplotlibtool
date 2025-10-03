#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import sys

import click
from asserttool import ic
from click_auto_help import AHGroup
from clicktool import CONTEXT_SETTINGS
from clicktool import click_add_options
from clicktool import click_global_options
from clicktool import tvicgvd
from configtool import get_config_directory
from eprint import eprint
from globalverbose import gvd
from PyQt6.QtWidgets import QApplication  # pylint: disable=E0611

from .dark_mode import enable_dark_mode
from .utils import load_points_from_stdin_ndarray

APP_NAME = "matplotlibtool"


@click.group(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    cls=AHGroup,
)
@click_add_options(click_global_options)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )
    config_directory = get_config_directory(click_instance=click, app_name=APP_NAME)
    config_directory.mkdir(exist_ok=True)
    ctx.obj["config_directory"] = config_directory


@cli.command()
@click.argument(
    "keys",
    type=str,
    nargs=-1,
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Enables normalization",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option(
    "--size",
    type=float,
    help="Point size",
)
@click.option(
    "--disable-antialiasing",
    is_flag=True,
    help="Disable antialiasing",
)
@click_add_options(click_global_options)
@click.pass_context
def plot2d(
    ctx: click.Context,
    keys: tuple[str, ...],
    normalize: bool,
    draw_lines: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    2D plot (Matplotlib): reads (x,y[,color]) tuples from stdin via messagepack.
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 2D points from stdin...")
    points_xyz = load_points_from_stdin_ndarray(minimum_dimensions=2)

    if points_xyz.shape[0] == 0:
        print("[ERROR] No valid 2D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Normalizing points to unit square"
        if normalize
        else "[INFO] Normalization disabled - centering only"
    )
    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib

    viewer = PointCloud2DViewerMatplotlib(
        points_xyz,
        normalize=normalize,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()


@cli.command()
@click.argument(
    "keys",
    type=str,
    nargs=-1,
)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable automatic normalization of point cloud coordinates (still centers at origin)",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option(
    "--xy",
    is_flag=True,
    help="Orthographic XY view",
)
@click.option(
    "--xz",
    is_flag=True,
    help="Orthographic XZ view",
)
@click.option(
    "--yz",
    is_flag=True,
    help="Orthographic YZ view",
)
@click.option(
    "--size",
    type=float,
    help="Point size (default: auto-detected based on point count)",
)
@click.option(
    "--disable-antialiasing",
    is_flag=True,
    help="Disable antialiasing for better performance with large datasets",
)
@click_add_options(click_global_options)
@click.pass_context
def plot3d(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    draw_lines: bool,
    xy: bool,
    xz: bool,
    yz: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    3D plot (Matplotlib): reads (x,y,z[,color]) tuples from stdin via messagepack.
    Features interactive 3D visualization with mouse controls and keyboard scaling.
    """

    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )
    view_mode = None
    if sum([xy, xz, yz]) > 1:
        print("[ERROR] Only one of --xy, --xz, or --yz may be specified.")
        sys.exit(1)
    if xy:
        view_mode = "xy"
    elif xz:
        view_mode = "xz"
    elif yz:
        view_mode = "yz"

    print("[INFO] Reading points from stdin...")
    points, color_data = load_points_from_stdin_for_3d()

    if points.shape[0] == 0:
        print("[ERROR] No valid points loaded. Exiting.")
        sys.exit(1)

    if disable_normalize:
        print(
            "[INFO] Normalization disabled - centering at origin but preserving scale"
        )
    else:
        print("[INFO] Normalizing points to unit cube")

    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    from .PointCloud3DViewerMatplotlib import PointCloud3DViewerMatplotlib

    viewer = PointCloud3DViewerMatplotlib(
        points,
        color_data=color_data,
        normalize=not disable_normalize,
        view_mode=view_mode,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()
