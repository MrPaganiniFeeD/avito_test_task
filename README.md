# avito_test_task
# Test Assignment for Avito — ML Pipeline

## 📌 Описание
Репозиторий содержит полный ML-пайплайн для решения задачи:
1. **Подготовка данных** — формирование датасета (`create_dataset.ipynb`)
2. **Обучение модели** — тренировка и сохранение модели (`train_model.ipynb`)
3. **Формирование сабмита** — генерация предсказаний для отправки (`create_submit.ipynb`)

---
## Общие рекомендации

- Запускайте ноутбуки в порядке: `create_dataset.ipynb` → `train_model.ipynb` → `create_submit.ipynb`.
- Убедитесь, что в папке `data/` лежат необходимые датасеты, о которых я написал ниже.
- Перед запуском установите зависимости: `pip install -r requirements.txt`.

	
- Для автоматического выполнения из терминала используется:
    
```bash
jupyter nbconvert --to notebook --execute create_dataset.ipynb --output create_dataset_out.ipynb
jupyter nbconvert --to notebook --execute train_model.ipynb --output train_model_out.ipynb
jupyter nbconvert --to notebook --execute create_submit.ipynb --output create_submit_out.ipynb
```

---
## 📂 Датасеты

Перед запуском необходимо скачать данные и поместить их в папку `data/`. Также в файле `create_dataset.ipynb` добавить пути к этим датасетам
Ещё нужно скачать тестовые данные `task_images` из задания и поместить в эту же папку `data/` 

- 📥 [Book detection2 (1157img)](https://universe.roboflow.com/online-detector/book-detection2/browse?queryText=split%3Atrain&pageSize=50&startingIndex=0&browseQuery=true)
- 📥 [All books (2070img)](https://universe.roboflow.com/zebra-learn/all-books-mumha/browse?queryText=class%3Abook+split%3Atrain&pageSize=50&startingIndex=150&browseQuery=true)
- 📥 [Book (200img)](https://universe.roboflow.com/kesiana-meco/book-2ivmo/browse?queryText=split%3Atrain&pageSize=50&startingIndex=0&browseQuery=true)
- 📥 [Book new (2155img)](https://universe.roboflow.com/a-fquda/book-new/dataset/13)
- 📥 [Book (2065img)](https://universe.roboflow.com/yrden/book-zbbr0/dataset/2)
- 📥 [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/code)
- 📥 [Book detection](https://universe.roboflow.com/slipernik/book-detection-lcl7n/dataset/2)
- 📥 [Book (1035img)](https://universe.roboflow.com/seopacme/book-m95oe/dataset/1)
- 📥 [Book(600img)](https://universe.roboflow.com/seopacme/book-m95oe/dataset/1)

## ⚙️ Структура

├── data/                            # Датасеты (скачать отдельно)
├── dataset/                       # Итоговый датасет
├── models/                       # Обученные модели
├── create_dataset.ipynb   # Подготовка данных
├── train_model.ipynb       # Обучение модели
├── create_submit.ipynb    # Генерация сабмита
├── requirements.txt          # Зависимости
├── run.sh                          # Скрипт для полного запуска
└── README.md                # Документация

## 🧠 Модель

- Архитектура: **yolo11m**
- Оптимизатор: **Adam** (lr=1e-3, weight_decay=1e-5)
- Аугментации: 
	- Цветовые аугментации → это добавляет устойчивость к свету/цвету
	- Пространственные аугментации → устойчивость к поворотам, масштабу и позиции
	- Микс-аугментации (mosaic/mixup) → разнообразие объектов. Смешивает изображения
- Обучение ведётся в `train_model.ipynb`
- Обученная модель сохраняется в `models/`

## 📊 Результаты (yolo11m, conf=0.5, iou=0.6)

| Метрика   | Значение |
| --------- | -------- |
| mAP50     | 0.665    |
| mAP50-95  | 0.584    |
| Precision | 0.814    |
| Recall    | 0.476    |
