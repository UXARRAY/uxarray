"""
File Purpose: defines helper functions to be used for testing optional dependencies.

The goal here is to spot-check that optional deps work as expected,
not to run an exhaustive set of tests with each combination of deps.
"""

def check_requires_no_opts():
    """run some checks which should not require any optional dependencies"""
    import uxarray as ux
    uxds = ux.tutorial.open_dataset('quad-hexagon')
    uxds.compute()

def check_requires_only_viz():
    """run some checks which should require viz optional dependencies,
    but not any other optional dependencies.
    """
    import uxarray as ux
    uxds = ux.tutorial.open_dataset('quad-hexagon')
    plot_obj = uxds.plot.points()  # points() doesn't need geo projection details.

    # actually try to render the plot, too:
    import holoviews as hv
    renderer = hv.renderer('matplotlib')
    renderer.get_plot(plot_obj)

def check_requires_only_geo():
    """run some checks which should require geo optional dependencies,
    but not any other optional dependencies.
    """
    import uxarray as ux
    arr = ux.tutorial.open_dataset('quad-hexagon')['t2m']
    arr.to_geodataframe()

def check_requires_viz_and_geo():
    """run some checks which should require both viz and geo optional dependencies,
    but not any other optional dependencies.
    """
    import uxarray as ux
    arr = ux.tutorial.open_dataset('quad-hexagon')['t2m']
    plot_obj = arr.plot.polygons()  # polygons() uses geo projection details.

    # actually try to render the plot, too:
    import holoviews as hv
    renderer = hv.renderer('matplotlib')
    renderer.get_plot(plot_obj)
