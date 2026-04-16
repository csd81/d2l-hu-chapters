# Szentimentelemzés és az adathalmaz
:label:`sec_sentiment`


Az online közösségi média és az értékelési platformok elterjedésével
rengeteg
véleménnyel teli adat keletkezett,
amelyek nagy potenciált hordoznak
a döntéshozatali folyamatok támogatása terén.
A *szentimentelemzés*
az emberek által előállított szövegekben
– mint például termékértékelések,
blogkommentek
és
fórumbeszélgetések –
meglévő érzelmeket tanulmányozza.
Széles körben alkalmazzák
olyan különféle területeken, mint
a politika (pl. a közvélemény elemzése szakpolitikákkal kapcsolatban),
a pénzügy (pl. a piaci hangulat elemzése)
és
a marketing (pl. termékkutatás és márkamenedzsment).

Mivel az érzelmek
diskrét polaritásokba vagy skálákra sorolhatók (pl. pozitív és negatív),
a szentimentelemzést
szövegosztályozási feladatnak tekinthetjük,
amely egy változó hosszúságú szöveges sorozatot
egy rögzített hosszúságú szövegkategóriává alakít át.
Ebben a fejezetben
a Stanford [nagy filmkritika-adathalmazát](https://ai.stanford.edu/%7Eamaas/data/sentiment/)
fogjuk használni szentimentelemzéshez.
Ez egy tanítóhalmazból és egy teszthalmazból áll,
mindkettő 25 000 IMDb-ről letöltött filmkritikát tartalmaz.
Mindkét adathalmazban
egyenlő számban szerepelnek
„pozitív" és „negatív" címkék,
amelyek különböző érzelmi polaritásokat jelölnek.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Az adathalmaz beolvasása

Először töltsük le és csomagoljuk ki az IMDb kritika-adathalmazt
a `../data/aclImdb` elérési útvonalra.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Ezután olvassuk be a tanító és a tesztelő adathalmazokat. Minden példa egy kritika és a hozzá tartozó címke: 1 a „pozitív" és 0 a „negatív" esetén.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Beolvassa az IMDb kritika-adathalmaz szöveges sorozatait és címkéit."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## Az adathalmaz előfeldolgozása

Az összes szót tokenként kezelve
és a 5-nél kevesebbszer előforduló szavakat kiszűrve
szótárt hozunk létre a tanítóadathalmazból.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

A tokenizálás után
rajzoljuk fel a kritikák hosszának hisztogramját
tokenek számában mérve.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Ahogy vártuk,
a kritikák különböző hosszúságúak.
Annak érdekében, hogy
ilyen kritikák egy minibatch-jét egyszerre tudjuk feldolgozni,
minden kritika hosszát 500-ra állítjuk csonkítással és párnázással,
ami hasonló a
gépi fordítási adathalmaznál alkalmazott
előfeldolgozási lépéshez
a :numref:`sec_machine_translation` fejezetben.

```{.python .input}
#@tab all
num_steps = 500  # sorozathossz
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Adatiterátorok létrehozása

Most létrehozhatjuk az adatiterátorokat.
Minden iterációban egy minibatch-nyi példa kerül visszaadásra.

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Összerakás

Végül az összes fenti lépést a `load_data_imdb` függvénybe foglaljuk.
Ez tanító és teszt adatiterátorokat, valamint az IMDb kritika-adathalmaz szótárát adja vissza.

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Visszaadja az adatiterátorokat és az IMDb kritika-adathalmaz szótárát."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Visszaadja az adatiterátorokat és az IMDb kritika-adathalmaz szótárát."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Összefoglalás

* A szentimentelemzés az emberek által előállított szövegekben lévő érzelmeket vizsgálja, amit szövegosztályozási feladatnak tekintünk, amely egy változó hosszúságú szöveges sorozatot rögzített hosszúságú szövegkategóriává alakít.
* Az előfeldolgozás után betölthetjük a Stanford nagy filmkritika-adathalmazát (IMDb kritika-adathalmaz) adatiterátorokba egy szótárral együtt.


## Feladatok


1. Milyen hiperparamétereket módosíthatunk ebben a részben a szentimentelemzési modellek tanításának felgyorsítása érdekében?
1. Meg tudnál valósítani egy függvényt, amely betölti az [Amazon értékelések](https://snap.stanford.edu/data/web-Amazon.html) adathalmazát adatiterátorokba és címkékbe szentimentelemzéshez?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:
