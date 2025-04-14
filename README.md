# Análise de Arremessos do Kobe Bryant - Projeto de Engenharia de Machine Learning

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Este projeto implementa uma análise de dados e modelo preditivo para os arremessos do Kobe Bryant durante sua carreira na NBA, utilizando o framework Kedro para organização e implementação.

## Diagrama do Pipeline

![alt text](docs/diagrama-geral.png)

## Ferramentas Utilizadas

- **Kedro**: framework para organização do projeto e pipelines
- **MLflow**: rastreamento de experimentos e versionamento de modelos
- **PyCaret**: AutoML e treinamento de modelos
- **Scikit-learn**: implementação dos modelos
- **Streamlit**: dashboard de monitoramento

Rastreamento de experimentos: MLflow registra automaticamente parâmetros , métricas e artefatos.
Funções de treinamento: Podem ser implementadas no Kedro, usando PyCaret ou sklearn.
Monitoramento da saúde do modelo: é feito via logs do MLflow e posteriormento no dashboard streamlit.
Atualização de modelo: Com base na performance no streamlit e MLflow, retreinamos usando o mesmo pipeline.
Provisionamento (Deployment): servimos o modelo como uma API via MLFlow.

## Artefatos gerados

No decorrer do projeto, vários artefatos foram criados:

1. Dados de entrada
- /data/01_raw/dataset_kobe_dev.parquet e /data/01_raw/dataset_kobe_prod.parquet.

2. Dados pré-processados
- /data/02_intermediate/data_filtered.parquet
- Dataset com as linhas nulas removidas e colunas selecionadas, conforme solicitado no enunciado.
- nome da run "data_processing"
- Log no MLflow

3. Bases de treino e teste
- /data/03_primary/base_train.parquet e /data/03_primary/base_test.parquet.
- split estratificado em 80/20.
- Também registramos no MLflow o tamanho de cada base, % de split

4. Modelos treinados
- modelos salvos em /data/06_models/
- nome da run "model_training"
- dois modelos:
  - Regressão Logística (com log loss)
  - Árvore de decisão (com log loss + F1 Score)

5. Previsões na base de produção
- Ao rodar o pipeline "inference", o script lê /data/01_raw/dataset_kobe_prod.parquet, carrega o modelo escolhido do MLflow e gera previsões, salvando em /data/07_model_output/
- log loss e F1 Score para a nova base.
- Nome da run no MLflow: "inference".
- O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.
  - O modelo treinado não é aderente a base de dados de produção. Pois os dados de produção são diferentes e todos os arremessos retornam 0.0% de predict.
- Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação.
  - Com variável resposta:
    - Podemos calcular métricas (log loss, F1) periodicamente, comparando com metas estabelecidas. Se cair muito => ALERTA.
  - Sem variável resposta:
    - Usamos técnicas de detecção de drift, como Population Stability Index (PSI) ou KL Divergence, etc.
    - Se a distribuição dos dados mudar muito, pode sinalizar perda de performance.
- Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.
  - Reativa
    - Esperar a métrica cair para só então disparar um retreinamento.
    - O modelo pode ficar defasado, mas é simples de implementar
  - Preditiva
    - Tenta antecipar a mudança no comportamento dos dados e agendar retreinamentos periodicamente
    - Registrar históricos de drifts, para um possível ajuste sazonal

6. Dashboard
- App streamlit
- Versionado em app/

## Como Executar

1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. Executar o pipeline:
```bash
kedro run
```

3. Servir o modelo via MLFlow:
```bash
mlflow models serve -m models:/treinamento_logistical_regression/1 --env-manager=local --port=5001
```

3. Iniciar a aplicação para novas inferências:
```bash
streamlit run app/main.py
```

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
