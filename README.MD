# seq2seq sorter

Попытка реализовать архитектуру Seq2Seq для задачи сортировки последовательности чисел на чистом питоне.  
Архитектура: one-hot embedding (где "-1" - go-тег) -> vanilla RNN encoder -> vanilla RNN decoder -> dense -> softmax.  
Минимизируемый функционал: кроссэнтропия  
Оптимизатор: SGD

Запуск:  
```
python3 seq.py
```

Гиперпараметры: batch size, epochs, learning rate

Думал, что надо уместить решение в текстовое поле на сайте, поэтому все компактно сделано, иначе более человечно бы написал.  
Трейн лосс падает, но качество все равно не очень - не хватило времени на lstm

