# Задание 1

В этом задании мы реализуем некоторые методы машинного обучения, которые помогут нам с реализацией нейросетей в следующих заданиях.

Перед выполнением задания:
- Запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки.
- Установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/).

### Часть 1
Метод К-ближайших соседей (K-neariest neighbor classifier)

`KNN.ipynb` - следуйте инструкциям в ноутбуке.

### Часть 2
Линейный классификатор (Linear classifier)

`Linear classifier.ipynb` - следуйте инструкциям в ноутбуке.

# Задание 2

В этом задании мы реализуем свою собственную нейронную сеть, а также научимся пользоваться [PyTorch](https://pytorch.org/) - одной из лучших библиотек для машинного обучения.

Перед выполнением задания:
- Запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки.
- Установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/).

### Часть 1
Нейронная сеть (Neural Network)

`Neural Network.ipynb` - следуйте инструкциям в ноутбуке.

### Часть 2
PyTorch

Это задание все еще можно выполнять без доступа к GPU.

`PyTorch.ipynb` - следуйте инструкциям в ноутбуке.

# Задание 3

В этом задании мы реализуем свою собственную сверточную нейронную сеть, сначала на numpy, а потом уже и на PyTorch.

Перед выполнением задания:
- Запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки.
- Установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/).

### Часть 1
Сверточная Нейронная Сеть (Convolutional Neural Network)

`CNN.ipynb` - следуйте инструкциям в ноутбуке.

### Часть 2
PyTorch CNN

Для этого задания уже требуется доступ к GPU.

Это может быть GPU от NVidia на вашем компьютере, тогда рекомендуется установить PyTorch с поддержкой GPU через Conda - https://pytorch.org/get-started/locally/

Если у вас нет GPU, можно воспользоваться [Google Colab](https://colab.research.google.com/), который предоставляет бесплатный доступ к GPU в облаке.

Туториал по настройке Google Colab:
https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
(Keras инсталлировать не нужно, наш notebook сам установит PyTorch)

`PyTorch_CNN.ipynb` - следуйте инструкциям в ноутбуке.

# Задание 4

В этом задании мы научимся пользоваться техниками transfer learning и fine-tuning на примере жизненной задачи распознавания хотдогов.

После выполнения этого задания у вас появится возможность поучавствовать в учебном соревновании Kaggle In-Class и сравнить свои результаты с другими участниками курса.

Участие - абсолютно опциональное и необязательное для завершения курса. Инструкции и детали - в конце ноутбука!

Для этого задания требуется доступ к GPU.

Это может быть GPU от NVidia на вашем компьютере, тогда рекомендуется установить PyTorch с поддержкой GPU через Conda - https://pytorch.org/get-started/locally/

Если у вас нет GPU, можно воспользоваться [Google Colab](https://colab.research.google.com/), который предоставляет бесплатный доступ к GPU в облаке.

Туториал по настройке Google Colab:
https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
(Keras инсталлировать не нужно, наш notebook сам установит PyTorch)

`HotDogOrNot.ipynb` - следуйте инструкциям в ноутбуке.

# Задание 5

В этом задании мы натренируем свои собственные word vectors двумя разными способами. Это задание можно делать на CPU.

Перед выполнением задания:
- Запустите файл `download_data.sh`, чтобы скачать данные, которые мы будем использовать для тренировки.
- Установите все необходимые библиотеки, запустив `pip install -r requirements.txt` (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/).

### Часть 1
Word2Vec

`Word2Vec.ipynb` - следуйте инструкциям в ноутбуке.

### Часть 2
Word2Vec with Negative Sampling

`Negative Sampling.ipynb` - следуйте инструкциям в ноутбуке.
