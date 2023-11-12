import plotly.io as pio

from .graph import *

pio.templates["kuarq"] = graph_template
pio.templates.default = "plotly+kuarq"
