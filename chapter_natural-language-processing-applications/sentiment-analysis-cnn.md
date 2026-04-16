# Szentimentelemzés: Konvolúciós neurális hálózatok alkalmazása
:label:`sec_sentiment_cnn` 


A :numref:`chap_cnn` fejezetben
kétdimenziós képadatok feldolgozásának mechanizmusait vizsgáltuk
kétdimenziós CNN-ek segítségével,
amelyeket helyi jellemzőkre, például szomszédos pixelekre alkalmaztunk.
Habár eredetileg
számítógépes látás céljára tervezték,
a CNN-eket természetes nyelvfeldolgozásra
is széles körben alkalmazzák.
Egyszerűen fogalmazva,
gondoljunk bármely szöveges sorozatra
egydimenziós képként.
Ily módon
az egydimenziós CNN-ek
képesek helyi jellemzőket feldolgozni,
például $n$-gramokat szövegben.

Ebben a részben
a *textCNN* modellt fogjuk használni annak bemutatására,
hogyan tervezzünk CNN architektúrát
egyedi szöveg reprezentálásához :cite:`Kim.2014`.
Összehasonlítva a :numref:`fig_nlp-map-sa-rnn` ábrával,
amely RNN architektúrát GloVe előtanítással használ
a szentimentelemzéshez,
az egyetlen különbség a :numref:`fig_nlp-map-sa-cnn` ábrán
az architektúra megválasztásában rejlik.


![Ez a rész az előtanított GloVe-ot CNN-alapú architektúrába táplálja szentimentelemzéshez.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## Egydimenziós konvolúciók

Mielőtt bemutatnánk a modellt,
nézzük meg, hogyan működik az egydimenziós konvolúció.
Ne feledjük, hogy ez csupán egy speciális esete
a kétdimenziós konvolúciónak,
amely a keresztkorreláció műveleten alapul.

![Egydimenziós keresztkorreláció-művelet. Az árnyékolt részek az első kimeneti elem, valamint a kimenet kiszámításához használt bemeneti és kernel-tensor elemek: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

Ahogyan a :numref:`fig_conv1d` ábrán látható,
az egydimenziós esetben
a konvolúciós ablak
balról jobbra csúszik
a bemeneti tensoron keresztül.
Csúszás közben
a konvolúciós ablakban egy adott pozícióban
lévő bemeneti résztensor (pl. $0$ és $1$ a :numref:`fig_conv1d` ábrán)
és a kernel-tensor (pl. $1$ és $2$ a :numref:`fig_conv1d` ábrán) elemenként összeszorozódnak.
Ezek a szorzatok összege
adja azt az egyetlen skaláris értéket (pl. $0\times1+1\times2=2$ a :numref:`fig_conv1d` ábrán),
amely a kimeneti tensor megfelelő pozíciójában áll.

Az egydimenziós keresztkorrelációt az alábbi `corr1d` függvénnyel valósítjuk meg.
A `X` bemeneti tensor
és a `K` kernel-tensor megadásával
a `Y` kimeneti tensort adja vissza.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

A :numref:`fig_conv1d` ábrából felépíthetjük a `X` bemeneti tensort és a `K` kernel-tensort, hogy ellenőrizzük a fenti egydimenziós keresztkorreláció implementáció kimenetét.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

Bármely
több csatornás egydimenziós bemenetnél
a konvolúciós kernelnek
ugyanannyi bemeneti csatornával kell rendelkeznie.
Ezután minden csatornánál
hajts végre keresztkorreláció-műveletet a bemenet egydimenziós tensorján és a konvolúciós kernel egydimenziós tensorján,
és az összes csatorna eredményét összegezve
állítsd elő az egydimenziós kimeneti tensort.
A :numref:`fig_conv1d_channel` ábra egy egydimenziós keresztkorreláció-műveletet mutat 3 bemeneti csatornával.

![Egydimenziós keresztkorreláció-művelet 3 bemeneti csatornával. Az árnyékolt részek az első kimeneti elem, valamint a kimenet kiszámításához használt bemeneti és kernel-tensor elemek: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`


Megvalósíthatjuk a több bemeneti csatornás egydimenziós keresztkorreláció-műveletet
és ellenőrizhetjük az eredményeket a :numref:`fig_conv1d_channel` ábrával.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # Először iterálunk az `X` és `K` 0-adik dimenzióján (csatorna dimenzió),
    # majd összeadjuk az eredményeket
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

Megjegyzendő, hogy
a több bemeneti csatornás egydimenziós keresztkorrelációk
egyenértékűek
az egybemeneti-csatornás
kétdimenziós keresztkorrelációkkal.
Szemléltetésként
a :numref:`fig_conv1d_channel` ábrán látható
több bemeneti csatornás egydimenziós keresztkorreláció
egyenértékű alakja
a :numref:`fig_conv1d_2d` ábrán látható
egybemeneti-csatornás
kétdimenziós keresztkorreláció,
ahol a konvolúciós kernel magasságának
meg kell egyeznie a bemeneti tensor magasságával.


![Kétdimenziós keresztkorreláció-művelet egyetlen bemeneti csatornával. Az árnyékolt részek az első kimeneti elem, valamint a kimenet kiszámításához használt bemeneti és kernel-tensor elemek: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

Mind a :numref:`fig_conv1d`, mind a :numref:`fig_conv1d_channel` kimenetei csupán egy csatornával rendelkeznek.
Csakúgy, mint a :numref:`subsec_multi-output-channels` részben leírt több kimeneti csatornás kétdimenziós konvolúcióknál,
több kimeneti csatornát is meghatározhatunk
egydimenziós konvolúcióknál.

## Időbeli maximum-pooling

Hasonlóképpen, poolingot alkalmazhatunk
a sorozatreprezentációkból a legmagasabb érték kinyerésére
mint a legfontosabb jellemző
az időlépések során.
A textCNN-ben használt *időbeli maximum-pooling*
úgy működik, mint
az egydimenziós globális maximális pooling
:cite:`Collobert.Weston.Bottou.ea.2011`.
Egy több csatornás bemenet esetén,
ahol minden csatorna különböző időlépéseken tárolja az értékeket,
az egyes csatornákon lévő kimenet
az adott csatorna maximális értéke.
Megjegyzendő, hogy
az időbeli maximum-pooling
különböző számú időlépést tesz lehetővé
különböző csatornákon.

## A textCNN modell

Az egydimenziós konvolúciót
és az időbeli maximum-poolingot alkalmazva
a textCNN modell
egyedi előtanított token-reprezentációkat vesz bemenetként,
majd sorozatreprezentációkat nyer és alakít át
a downstream alkalmazáshoz.

Egy $n$ tokenből álló egyedi szöveges sorozat esetén,
amelyet $d$-dimenziós vektorok képviselnek,
a bemeneti tensor szélessége, magassága és csatornáinak száma
rendre $n$, $1$ és $d$.
A textCNN modell a következőképpen alakítja át a bemenetet kimenetté:

1. Definiálj több egydimenziós konvolúciós kernelt, és hajts végre konvolúciós műveleteket külön-külön a bemeneteken. A különböző szélességű konvolúciós kernelek különböző számú szomszédos token közötti helyi jellemzőket ragadhatnak meg.
1. Hajts végre időbeli maximum-poolingot az összes kimeneti csatornán, majd fűzd össze az összes skaláris pooling-kimenetet vektorrá.
1. Alakítsd át az összefűzött vektort kimeneti kategóriákká a teljesen összekötött réteggel. Dropout alkalmazható a túlilleszkedés csökkentésére.

![A textCNN modell architektúrája.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

A :numref:`fig_conv1d_textcnn` ábra
egy konkrét példával szemlélteti a textCNN modell architektúráját.
A bemenet egy 11 tokenből álló mondat,
ahol
minden token 6-dimenziós vektorokkal van reprezentálva.
Tehát egy 6 csatornás bemenetet kapunk 11-es szélességgel.
Definiáljunk
két egydimenziós konvolúciós kernelt
2-es és 4-es szélességgel,
4 és 5 kimeneti csatornával.
Ezek
4 kimeneti csatornát állítanak elő $11-2+1=10$ szélességgel
és 5 kimeneti csatornát $11-4+1=8$ szélességgel.
Annak ellenére, hogy e 9 csatorna különböző szélességű,
az időbeli maximum-pooling
összefűzött 9-dimenziós vektort ad,
amelyet végül
2-dimenziós kimeneti vektorrá alakítanak
bináris szentimentjósláshoz.



### A modell definiálása

A textCNN modellt az alábbi osztályban valósítjuk meg.
A :numref:`sec_sentiment_rnn` fejezetben szereplő kétirányú RNN modellhez képest,
a rekurzív rétegek konvolúciós rétegekkel való felváltása mellett
két embedding réteget is alkalmazunk:
az egyiket tanítható súlyokkal, a másikat
rögzített súlyokkal.

```{.python .input}
#@tab mxnet
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Az embedding réteg nem kerül tanításra
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # Az időbeli maximum-pooling rétegnek nincsenek paraméterei,
        # ezért ez a példány megosztható
        self.pool = nn.GlobalMaxPool1D()
        # Több egydimenziós konvolúciós réteg létrehozása
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Két embedding réteg kimenetének összefűzése (batch méret, tokenek
        # száma, token vektor dimenzió) alakban, vektorok mentén
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Az egydimenziós konvolúciós rétegek bemeneti formátumának megfelelően
        # átrendezzük a tensort, hogy a második dimenzió tárolja a csatornákat
        embeddings = embeddings.transpose(0, 2, 1)
        # Minden egydimenziós konvolúciós rétegnél az időbeli maximum-pooling
        # után (batch méret, csatornák száma, 1) alakú tensor keletkezik.
        # Eltávolítjuk az utolsó dimenziót és összefűzzük csatornák mentén
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Az embedding réteg nem kerül tanításra
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # Az időbeli maximum-pooling rétegnek nincsenek paraméterei,
        # ezért ez a példány megosztható
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Több egydimenziós konvolúciós réteg létrehozása
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Két embedding réteg kimenetének összefűzése (batch méret, tokenek
        # száma, token vektor dimenzió) alakban, vektorok mentén
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Az egydimenziós konvolúciós rétegek bemeneti formátumának megfelelően
        # átrendezzük a tensort, hogy a második dimenzió tárolja a csatornákat
        embeddings = embeddings.permute(0, 2, 1)
        # Minden egydimenziós konvolúciós rétegnél az időbeli maximum-pooling
        # után (batch méret, csatornák száma, 1) alakú tensor keletkezik.
        # Eltávolítjuk az utolsó dimenziót és összefűzzük csatornák mentén
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

Hozzunk létre egy textCNN példányt.
3 konvolúciós réteggel rendelkezik, amelyek kernel-szélességei 3, 4 és 5, mindegyik 100 kimeneti csatornával.

```{.python .input}
#@tab mxnet
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights);
```

### Előtanított szóvektorok betöltése

Ahogy a :numref:`sec_sentiment_rnn` fejezetben,
betöltjük az előtanított 100-dimenziós GloVe embeddingeket
az inicializált token-reprezentációkként.
Ezek a token-reprezentációk (embedding-súlyok)
az `embedding`-ben kerülnek betanításra,
míg a `constant_embedding`-ben rögzítve maradnak.

```{.python .input}
#@tab mxnet
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### A modell tanítása és kiértékelése

Most betaníthatjuk a textCNN modellt a szentimentelemzéshez.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Az alábbiakban a betanított modellt használjuk két egyszerű mondat szentimentjének előrejelzésére.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## Összefoglalás

* Az egydimenziós CNN-ek képesek helyi jellemzőket feldolgozni, például $n$-gramokat szövegben.
* A több bemeneti csatornás egydimenziós keresztkorrelációk egyenértékűek az egybemeneti-csatornás kétdimenziós keresztkorrelációkkal.
* Az időbeli maximum-pooling különböző számú időlépést tesz lehetővé különböző csatornákon.
* A textCNN modell egyedi token-reprezentációkat alakít át downstream alkalmazás kimenetekké egydimenziós konvolúciós rétegek és időbeli maximum-pooling rétegek segítségével.


## Feladatok

1. Hangold a hiperparamétereket, és hasonlítsd össze a két architektúrát a szentimentelemzéshez a :numref:`sec_sentiment_rnn` fejezetben és ebben a részben, például az osztályozási pontosság és a számítási hatékonyság tekintetében.
1. Tovább javítható-e a modell osztályozási pontossága a :numref:`sec_sentiment_rnn` fejezet feladataiban bemutatott módszerekkel?
1. Adj hozzá pozíciókódolást a bemeneti reprezentációkhoz. Javítja-e az osztályozási pontosságot?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1425)
:end_tab:
