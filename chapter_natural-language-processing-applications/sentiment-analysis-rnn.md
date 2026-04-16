# Szentimentelemzés: Rekurzív neurális hálózatok alkalmazása
:label:`sec_sentiment_rnn` 


A szóhasonlóság és analógia feladatokhoz hasonlóan
az előtanított szóvektorokat
szentimentelemzésre is alkalmazhatjuk.
Mivel az IMDb kritika-adathalmaz
a :numref:`sec_sentiment` fejezetben
nem túl nagy,
a nagyszabású korpuszon
előtanított szöveges reprezentációk használata
csökkenthet a modell túlilleszkedésén.
Konkrét példaként,
ahogyan a :numref:`fig_nlp-map-sa-rnn` ábra szemlélteti,
minden tokent
az előtanított GloVe modell segítségével fogunk reprezentálni,
és ezeket a token-reprezentációkat
egy többrétegű kétirányú RNN-be tápláljuk
a szöveges sorozat-reprezentáció megszerzéséhez,
amelyet majd
szentimentelemzési kimenetekké alakítunk :cite:`Maas.Daly.Pham.ea.2011`.
Ugyanennél a downstream alkalmazásnál
később egy eltérő architektúrális megoldást is fogunk vizsgálni.

![Ez a rész az előtanított GloVe-ot RNN-alapú architektúrába táplálja szentimentelemzéshez.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Egyedi szöveg reprezentálása RNN-ekkel

A szövegosztályozási feladatokban,
mint például a szentimentelemzésben,
egy változó hosszúságú szöveges sorozat
rögzített hosszúságú kategóriákká alakul.
A következő `BiRNN` osztályban,
miközben a szöveges sorozat minden tokenje
egyéni
előtanított GloVe
reprezentációt kap az embedding rétegen keresztül
(`self.embedding`),
a teljes sorozatot
egy kétirányú RNN kódolja (`self.encoder`).
Konkrétabban,
a kétirányú LSTM rejtett állapotai (az utolsó rétegben)
mind a kezdő, mind a végső időlépésnél
összefűzésre kerülnek
a szöveges sorozat reprezentációjaként.
Ezt az egyedi szöveges reprezentációt
ezután kimeneti kategóriákká alakítja
egy teljesen összekötött réteg (`self.decoder`)
két kimenettel („pozitív" és „negatív").

```{.python .input}
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # A `bidirectional` True értékre állításával kétirányú RNN-t kapunk
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # Az `inputs` alakja (kötegméret, időlépések száma). Mivel az LSTM
        # megköveteli, hogy a bemenet első dimenziója az időbeli dimenzió
        # legyen, a bemenetet transzponáljuk a token-reprezentációk
        # megszerzése előtt. A kimenet alakja (időlépések száma, kötegméret,
        # szóvektor-dimenzió)
        embeddings = self.embedding(inputs.T)
        # Az utolsó rejtett réteg rejtett állapotait adja vissza különböző
        # időlépéseknél. Az `outputs` alakja (időlépések száma, kötegméret,
        # 2 * rejtett egységek száma)
        outputs = self.encoder(embeddings)
        # A kezdő és végső időlépések rejtett állapotainak összefűzése
        # a teljesen összekötött réteg bemeneteként. Alakja (kötegméret,
        # 4 * rejtett egységek száma)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # A `bidirectional` True értékre állításával kétirányú RNN-t kapunk
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # Az `inputs` alakja (kötegméret, időlépések száma). Mivel az LSTM
        # megköveteli, hogy a bemenet első dimenziója az időbeli dimenzió
        # legyen, a bemenetet transzponáljuk a token-reprezentációk
        # megszerzése előtt. A kimenet alakja (időlépések száma, kötegméret,
        # szóvektor-dimenzió)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Az utolsó rejtett réteg rejtett állapotait adja vissza különböző
        # időlépéseknél. Az `outputs` alakja (időlépések száma, kötegméret,
        # 2 * rejtett egységek száma)
        outputs, _ = self.encoder(embeddings)
        # A kezdő és végső időlépések rejtett állapotainak összefűzése
        # a teljesen összekötött réteg bemeneteként. Alakja (kötegméret,
        # 4 * rejtett egységek száma)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

Hozzunk létre egy két rejtett rétegű kétirányú RNN-t az egyedi szöveg reprezentálásához a szentimentelemzéshez.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## Előtanított szóvektorok betöltése

Az alábbiakban betöltjük az előtanított 100-dimenziós (összhangban kell lennie az `embed_size`-zal) GloVe embeddingeket a szótárban lévő tokenekhez.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

Nyomtassuk ki a szótárban lévő összes token vektorainak alakját.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Ezeket az előtanított
szóvektorokat használjuk
a kritikákban lévő tokenek reprezentálásához,
és tanítás közben
nem frissítjük ezeket a vektorokat.

```{.python .input}
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## A modell tanítása és kiértékelése

Most betaníthatjuk a kétirányú RNN-t szentimentelemzéshez.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Az alábbi függvényt definiáljuk a szöveges sorozat szentimentjének előrejelzéséhez a betanított `net` modell segítségével.

```{.python .input}
#@tab mxnet
#@save
def predict_sentiment(net, vocab, sequence):
    """Egy szöveges sorozat szentimentjének előrejelzése."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Egy szöveges sorozat szentimentjének előrejelzése."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Végül a betanított modellt két egyszerű mondat szentimentjének előrejelzésére alkalmazzuk.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Összefoglalás

* Az előtanított szóvektorok képesek egy szöveges sorozat egyedi tokenjeit reprezentálni.
* A kétirányú RNN-ek képesek egy szöveges sorozatot reprezentálni, például a kezdő és végső időlépések rejtett állapotainak összefűzésével. Ez az egyedi szöveges reprezentáció kategóriákká alakítható egy teljesen összekötött réteg segítségével.



## Feladatok

1. Növeld az epochok számát. Javítható-e a tanítási és tesztelési pontosság? Mi a helyzet más hiperparaméterek hangolásával?
1. Használj nagyobb előtanított szóvektorokat, például 300-dimenziós GloVe embeddingeket. Javítja-e az osztályozási pontosságot?
1. Javítható-e az osztályozási pontosság a spaCy tokenizáció alkalmazásával? Telepíteni kell a spaCy-t (`pip install spacy`) és az angol csomagot (`python -m spacy download en`). A kódban először importáld a spaCy-t (`import spacy`). Ezután töltsd be a spaCy angol csomagot (`spacy_en = spacy.load('en')`). Végül definiáld a következő függvényt: `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`, és helyettesítsd be az eredeti `tokenizer` függvényt. Figyeld meg a kifejezés tokenek különböző formáit a GloVe-ban és a spaCy-ban. Például a „new york" kifejezés tokenje a GloVe-ban „new-york" formában, míg a spaCy tokenizálás után „new york" formában szerepel.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:
