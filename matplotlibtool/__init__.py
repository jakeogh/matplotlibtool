"""
The package imports nothing, so a module that wants only the lane
arithmetic in Plot2DOverlay does not pull in matplotlib's Qt backend to get
it. The viewer is `from matplotlibtool.Plot2D import Plot2D`, and its group
context `from matplotlibtool.PlotGroupContext import PlotGroupContext`.

Neither class is offered on the package: each shares its module's name, and
importing a module binds the module to that name on the package, so a
package-level `Plot2D` would be the module or the class depending on what
had been imported before, and `from matplotlibtool import Plot2D` the
module every time.
"""
