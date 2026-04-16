# Transzponált konvolúció
:label:`sec_transposed_conv`

Az eddig látott CNN-rétegek,
mint a konvolúciós rétegek (:numref:`sec_conv_layer`) és a pooling rétegek (:numref:`sec_pooling`),
jellemzően csökkentik (lemintavételezik) a bemenet térbeli dimenzióit (magasságát és szélességét),
vagy változatlanul hagyják azokat.
A szemantikus szegmentálásban,
amely pixel szinten osztályoz,
kényelmes lenne,
ha a bemenet és a kimenet térbeli dimenziói megegyeznének.
Például
egy kimeneti pixel csatorna-dimenziója
tárolhatja az azonos térbeli pozíciójú bemeneti pixel osztályozási eredményeit.

Ennek elérése érdekében – különösen azután, hogy a CNN-rétegek csökkentették a térbeli dimenziókat –
egy másik típusú CNN-réteget alkalmazhatunk,
amely növelni (felfelé mintavételezni) képes
a közbülső jellemzőtérképek térbeli dimenzióit.
Ebben a szakaszban bemutatjuk
a *transzponált konvolúciót*, amelyet *törtrészes lépéses konvolúciónak* is neveznek :cite:`Dumoulin.Visin.2016`,
amely a konvolúció által végzett lemintavételezési műveletek megfordítására szolgál.

```{.python .input}
#@tab mxnet
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Alapművelet

A csatornáktól egyelőre eltekintve
kezdjük az alapvető transzponált konvolúciós művelettel,
amelynek lépésköze 1 és nincs kitöltés.
Tegyük fel, hogy adott egy
$n_h \times n_w$ méretű bemeneti tenzor
és egy $k_h \times k_w$ méretű kernel.
A kernel ablakát 1-es lépésközzel $n_w$-szer tolva minden sorban
és $n_h$-szor minden oszlopban
összesen $n_h n_w$ közbenső eredményt kapunk.
Minden közbenső eredmény
egy nullákkal inicializált $(n_h + k_h - 1) \times (n_w + k_w - 1)$
méretű tenzor.
Minden közbenső tenzor kiszámításához
a bemeneti tenzor minden elemét megszorozzuk a kernellel,
és az eredményül kapott $k_h \times k_w$ tenzorral
a közbenső tenzor egy részét felülírjuk.
Fontos, hogy a felülírt rész helyzete a közbenső tenzorban
megfelel a számításhoz használt bemeneti tenzorbeli elem pozíciójának.
Végül az összes közbenső eredményt összeadjuk, hogy megkapjuk a kimenetet.

Példaképpen
a :numref:`fig_trans_conv` ábra szemlélteti,
hogyan számítják ki a transzponált konvolúciót $2\times 2$-es kernellel egy $2\times 2$-es bemeneti tenzorra.


![Transzponált konvolúció $2\times 2$-es kernellel. Az árnyékolt részek egy közbenső tenzor részei, valamint a számításhoz felhasznált bemeneti és kerneltenzor elemei.](../img/trans_conv.svg)
:label:`fig_trans_conv`


(**Megvalósíthatjuk ezt az alapvető transzponált konvolúciós műveletet**) `trans_conv` formájában egy `X` bemeneti mátrixra és egy `K` kernelmátrixra.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

A szokásos konvolúcióval ellentétben (lásd :numref:`sec_conv_layer`), amely a kernelen keresztül *csökkenti* a bemeneti elemeket,
a transzponált konvolúció
*szórja* a bemeneti elemeket a kernelen keresztül,
ezáltal a bemenetnél nagyobb kimenetet hozva létre.
A :numref:`fig_trans_conv` ábrából felépíthetjük az `X` bemeneti tenzort és a `K` kerneltenzort, hogy [**ellenőrizzük a fenti megvalósítás kimenetét**] az alap kétdimenziós transzponált konvolúciós műveletnél.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

Ha az `X` bemenet és a `K` kernel egyaránt
négydimenziós tenzorok,
[**magas szintű API-k segítségével ugyanazokat az eredményeket kaphatjuk**].

```{.python .input}
#@tab mxnet
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**Kitöltés, lépésköz és több csatorna**]

A szokásos konvolúcióval ellentétben, ahol a kitöltés a bemenetre vonatkozik,
a transzponált konvolúcióban a kimenetre alkalmazzuk.
Például ha a magasság és szélesség mindkét oldalán
1-es kitöltést adunk meg,
az első és az utolsó sor, illetve oszlop
eltávolításra kerül a transzponált konvolúció kimenetéből.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

A transzponált konvolúcióban a lépésközök a közbenső eredményekre (azaz a kimenetre) vonatkoznak, nem a bemenetre.
A :numref:`fig_trans_conv` ábrából kiindulva, ugyanazon bemeneti és kerneltenzorokat használva,
a lépésköz 1-ről 2-re változtatása
megnöveli a közbenső tenzorok magasságát és szélességét,
így a kimenet is nagyobb lesz (:numref:`fig_trans_conv_stride2`).


![Transzponált konvolúció $2\times 2$-es kernellel, 2-es lépésközzel. Az árnyékolt részek egy közbenső tenzor részei, valamint a számításhoz felhasznált bemeneti és kerneltenzor elemei.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`



Az alábbi kódrészlet ellenőrzi a :numref:`fig_trans_conv_stride2` ábrában szereplő 2-es lépésközű transzponált konvolúció kimenetét.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

Több bemeneti és kimeneti csatorna esetén
a transzponált konvolúció ugyanúgy működik, mint a szokásos konvolúció.
Tegyük fel, hogy
a bemenetnek $c_i$ csatornája van,
és a transzponált konvolúció
minden bemeneti csatornához egy $k_h\times k_w$ kerneltenzort rendel.
Ha több kimeneti csatornát adunk meg,
minden kimeneti csatornához egy $c_i\times k_h\times k_w$ méretű kernelünk lesz.


Összességében ha az $\mathsf{X}$-et egy $f$ konvolúciós rétegbe tápláljuk és a kimenet $\mathsf{Y}=f(\mathsf{X})$, majd létrehozunk egy $g$ transzponált konvolúciós réteget az $f$-fel megegyező hiperparaméterekkel, kivéve, hogy a kimeneti csatornák száma az $\mathsf{X}$-ben lévő csatornák számával egyezik meg, akkor $g(Y)$ alakja megegyezik $\mathsf{X}$ alakjával.
Ezt az alábbi példa szemlélteti.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**Kapcsolat a mátrixtranszpozícióval**]
:label:`subsec-connection-to-mat-transposition`

A transzponált konvolúció a mátrixtranszpozícióról kapta a nevét.
Ennek magyarázatához nézzük meg először,
hogyan valósíthatók meg a konvolúciók mátrixszorzással.
Az alábbi példában definiálunk egy $3\times 3$-as `X` bemenetet és egy $2\times 2$-es `K` konvolúciós kernelt, majd a `corr2d` függvénnyel kiszámítjuk a `Y` konvolúciós kimenetet.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

Ezután a `K` konvolúciós kernelt átírjuk egy sok nullát tartalmazó
`W` ritka súlymátrixszá.
A súlymátrix alakja ($4$, $9$),
ahol a nem nulla elemek a `K` konvolúciós kernelből származnak.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

Az `X` bemenetet soronként összefűzve egy 9 hosszúságú vektort kapunk. Ekkor a `W` és a vektorizált `X` mátrixszorzata 4 hosszúságú vektort ad.
Átformálás után ugyanazt a `Y` eredményt kapjuk,
mint a fenti eredeti konvolúciós műveletnél:
mátrixszorzással valósítottuk meg a konvolúciót.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

Hasonlóképpen megvalósítható a transzponált konvolúció mátrixszorzással.
A következő példában
a fenti szokásos konvolúció $2 \times 2$-es `Y` kimenetét
transzponált konvolúció bemeneteként használjuk.
Ennek a műveletnek a mátrixszorzással való megvalósításához
csupán a $(9, 4)$ méretű `W` súlymátrixot kell transzponálni.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

Gondoljunk a konvolúció mátrixszorzással való megvalósítására.
Adott egy $\mathbf{x}$ bemeneti vektor
és egy $\mathbf{W}$ súlymátrix,
a konvolúció előremenő propagálási függvénye
megvalósítható a bemenet és a súlymátrix szorzatával,
amely $\mathbf{y}=\mathbf{W}\mathbf{x}$ vektort ad eredményül.
Mivel a visszaterjesztés
a láncszabályt követi
és $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$,
a konvolúció visszaterjesztési függvénye
megvalósítható a bemenetnek a transzponált $\mathbf{W}^\top$ súlymátrixszal való szorzatával.
Ebből következően
a transzponált konvolúciós réteg
egyszerűen felcseréli a konvolúciós réteg előremenő propagálási
és visszaterjesztési függvényét:
az előremenő propagálási és visszaterjesztési függvényei
a bemeneti vektort
rendre $\mathbf{W}^\top$-vel és $\mathbf{W}$-vel szorozzák.


## Összefoglalás

* A szokásos konvolúcióval ellentétben, amely a kernelen keresztül csökkenti a bemeneti elemeket, a transzponált konvolúció szórja a bemeneti elemeket a kernelen keresztül, ezáltal a bemenetnél nagyobb kimenetet hozva létre.
* Ha az $\mathsf{X}$-et egy $f$ konvolúciós rétegbe tápláljuk és a kimenet $\mathsf{Y}=f(\mathsf{X})$, majd létrehozunk egy $g$ transzponált konvolúciós réteget az $f$-fel megegyező hiperparaméterekkel, kivéve, hogy a kimeneti csatornák száma az $\mathsf{X}$-ben lévő csatornák számával egyezik meg, akkor $g(Y)$ alakja megegyezik $\mathsf{X}$ alakjával.
* A konvolúciók megvalósíthatók mátrixszorzással. A transzponált konvolúciós réteg egyszerűen felcseréli a konvolúciós réteg előremenő propagálási és visszaterjesztési függvényét.


## Feladatok

1. A :numref:`subsec-connection-to-mat-transposition` szakaszban a `X` konvolúciós bemenet és a transzponált konvolúció `Z` kimenete azonos alakú. Értékeik is megegyeznek? Miért?
1. Hatékony-e mátrixszorzással megvalósítani a konvolúciókat? Miért?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
