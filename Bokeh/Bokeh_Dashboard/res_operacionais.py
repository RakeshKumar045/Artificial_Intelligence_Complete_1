import numbers
from datetime import datetime, date

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, layout
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models import HoverTool
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.widgets import DateRangeSlider, Select, Button, Div, CheckboxGroup
from bokeh.plotting import figure
from generate_data import generate_df

dataset = generate_df()  # original
dataset['Data'] = [i.to_pydatetime().date() for i in dataset['Data']]
current_dataset = dataset.copy()  # filtered by widgets

# ==================================================
# Main plot
# ==================================================

# Defining the data
anos = ['2017', '2018']
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
         'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

options = ["Vendas", "Margem_Operacional", "EBITDA"]
colors = ['#c6dbef', '#6baed6', '#084594']
colormapper = dict(zip(options, colors))
mapa_produtos = {0: "Produto_A", 1: "Produto_B", 2: "Produto_C"}

source = ColumnDataSource(data=dict(x=[], y=[], labels=[], color=[]))
# initialize widgets
select = Select(value="Vendas", options=options)
dates = DateRangeSlider(start=date(2017, 1, 1),
                        end=date(2018, 9, 1),
                        value=(date(2017, 1, 1), date(2018, 9, 1)), step=1)
produtos = CheckboxGroup(labels=["Produto_A", "Produto_B", "Produto_C"],
                         active=[0, 1, 2])
div1 = Div(text="<br><h5>Medida</h5>")
div2 = Div(text="<br><h5>Datas</h5>")
div3 = Div(text="<br><h5>Produtos</h5>")
hover = HoverTool(tooltips=[
    ("Valores", "@y{0.0} m€"),
    ("Mês", "@x")],
    formatters={
        'Mês': 'printf',
    },
    mode='vline'
)


def refresh_data(dataset):
    """Refreshes the source.data based on the dataset provided"""
    if isinstance(dates.value[0], numbers.Integral):
        start_date = datetime.fromtimestamp(dates.value[0] / 1000).date()
        end_date = datetime.fromtimestamp(dates.value[1] / 1000).date()
    else:
        start_date, end_date = dates.value
    # Apply filters
    dataset_filtered = dataset[(dataset['Data'] >= start_date) & (dataset['Data'] <= end_date)]
    lista_produtos = [mapa_produtos[i] for i in produtos.active]
    dataset_filtered = dataset_filtered[dataset_filtered['Produto'].isin(lista_produtos)]
    # Apply calculations
    dataset_filtered = dataset_filtered.groupby('Ano/Mês')[select.value].sum().reset_index()
    x = dataset_filtered['Ano/Mês']
    y = dataset_filtered[select.value]
    labels = ["{}m €".format(int(i)) for i in y]
    color = [colormapper[select.value] for i in y]
    source.data = {'x': x,
                   'y': y,
                   'labels': labels,
                   'color': color}


refresh_data(dataset)  # initialize data

# Creating the figure
x_range = [(ano, mes) for ano in anos for mes in meses]
x_range = x_range[:len(x_range) - 4]  # remove last 4 months, no data

plot = figure(x_range=FactorRange(*x_range), y_range=(0, max(source.data['y']) * 1.2),
              title=select.value,
              tools=[hover],
              plot_height=500, plot_width=800,
              sizing_mode='stretch_both',
              toolbar_location=None)

plot.vbar(x='x', top='y', width=0.7, source=source, fill_color='color')


def generate_data():
    """Generates new data on button click"""
    dataset = generate_df()
    refresh_data(dataset=dataset)


button = Button(label="Gerar novos dados", button_type="primary")
button.on_click(generate_data)


def update_medida(attrname, old, new):
    """Updates medida being used on select change"""
    plot.title.text = select.value
    refresh_data(dataset)


select.on_change('value', update_medida)


def update_date(attrname, old, new):
    """Updates data on DateRangeSlider change"""
    start_date = datetime.fromtimestamp(dates.value[0] / 1000).date()
    end_date = datetime.fromtimestamp(dates.value[1] / 1000).date()
    dataset_filtered = dataset.copy()
    dataset_filtered = dataset_filtered[
        (dataset_filtered['Data'] >= start_date) & (dataset_filtered['Data'] <= end_date)]
    refresh_data(dataset_filtered)


dates.on_change('value', update_date)


def update_produtos(attrname, old, new):
    """Updates Produtos on CheckboxGroup change"""
    lista_produtos = [mapa_produtos[i] for i in produtos.active]
    dataset_filtered = dataset.copy()
    dataset_filtered = dataset_filtered[dataset_filtered['Produto'].isin(lista_produtos)]
    refresh_data(dataset_filtered)


produtos.on_change('active', update_produtos)

# Define static layouts
plot.sizing_mode = 'scale_width'
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None

plot.outline_line_color = None
plot.yaxis.minor_tick_line_color = None
plot.yaxis.axis_label = "Valor em m€"
plot.yaxis[0].formatter = NumeralTickFormatter(format="€ 0,0")

controls = [button, div1, select, div2, dates, div3, produtos]

inputs = widgetbox(*controls, width=300, height=200, sizing_mode='fixed')
plot_layout = layout([
    [plot, inputs],
])

curdoc().add_root(plot_layout)
