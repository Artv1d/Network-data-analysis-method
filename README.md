# Network-data-analysis-method
# Pipeline обработки данных с цепочкой ответственности

Проект для обработки CSV файлов с использованием паттерна проектирования "Цепочка ответственности".

## Установка

```bash
pip install -e .```

## Использование

```bash
python -m src.app путь/к/hh.csv --target "ЗП"```


## Структура

src/app.py - точка входа командной строки

src/pipeline/base_handler.py - абстрактный базовый класс обработчика

src/pipeline/handlers.py - конкретные реализации обработчиков

src/pipeline/data_processor.py - координатор цепочки обработки

## Выходные файлы

x_data.npy - матрица признаков

y_data.npy - вектор целевой переменной

feature_names.txt - имена признаков

processing_report.txt - отчет об обработке

