from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from qcc.file import load_toml

from .graph import *

CWD = Path(__file__).parent
graph_template = go.layout.Template(load_toml(CWD / "graph_template.toml"))
pio.templates["kuarq"] = graph_template
pio.templates.default = "plotly+kuarq"
