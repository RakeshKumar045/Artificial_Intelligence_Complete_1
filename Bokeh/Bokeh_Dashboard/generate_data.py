import random
from datetime import date

import numpy as np
import pandas as pd


def generate_df():
    """Genera dados para serem usados no dashboard Bokeh"""

    SAMPLE_SIZE = 500
    lista_meses = np.arange(1, 13)
    lista_anos = [2017, 2018]
    lista_produtos = ['Produto_A', 'Produto_B', 'Produto_C']

    day = [random.choice(np.arange(1, 29)) for i in range(SAMPLE_SIZE)]
    month = [random.choice(lista_meses) for i in range(SAMPLE_SIZE)]
    year = [random.choice(lista_anos) for i in range(SAMPLE_SIZE)]

    vendas = [random.triangular(20, 200, 50) for i in range(SAMPLE_SIZE)]
    custos_diretos = [vendas[i] * random.triangular(0.2, 0.6, 0.3) for i in range(SAMPLE_SIZE)]
    produto = [random.choice(lista_produtos) for i in range(SAMPLE_SIZE)]
    margem_op = [vendas[i] - custos_diretos[i] for i in range(SAMPLE_SIZE)]
    performance_eng = [random.normalvariate(0.9, 0.5) for i in range(SAMPLE_SIZE)]
    performance_eng = [i if i <= 1 else 1 for i in performance_eng]
    custo_aquisicao_cliente = [vendas[i] * random.triangular(0.1, 0.4, 0.15) for i in range(SAMPLE_SIZE)]
    margem_bruta = [vendas[i] - custos_diretos[i] - custo_aquisicao_cliente[i] for i in range(SAMPLE_SIZE)]
    distribuicao_otd = [1] * 12 + [0] * 2
    on_time_delivery = [random.choice(distribuicao_otd) for i in range(SAMPLE_SIZE)]

    dates = pd.DataFrame(
        {
            'day': day,
            'month': month,
            'year': year,
        }
    )

    dates = pd.to_datetime(dates)
    dates.to_frame()

    dataset = pd.DataFrame(
        {
            'Vendas': vendas,
            'Custos_Diretos': custos_diretos,
            'Produto': produto,
            'Margem_Operacional': margem_op,
            'Performance_Engenharia': performance_eng,
            'Custo_Aquisicao_Cliente': custo_aquisicao_cliente,
            'EBITDA': margem_bruta,
            'OnTimeDelivery': on_time_delivery,
        }
    )
    dataset = pd.concat([dates, dataset], axis=1)
    dataset = dataset.rename(columns={0: 'Data'})
    dataset = dataset[dataset['Data'] <= date(2018, 9, 1)]

    meses = ['Unknown', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    dataset['Mês'] = [meses[i.month] for i in dataset.Data]
    dataset['Ano'] = [i.year for i in dataset.Data]
    dataset['Ano/Mês'] = [(str(i), str(j)) for i, j in zip(dataset['Ano'], dataset['Mês'])]
    return dataset
