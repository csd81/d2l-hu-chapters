```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# A Kódoló–Dekódoló Architektúra
:label:`sec_encoder-decoder`

Az általános sorozatból sorozatba irányuló problémáknál,
mint amilyen a gépi fordítás
(:numref:`sec_machine_translation`),
a bemenetek és kimenetek változó hosszúságú,
nem igazított sorozatok.
Az ilyen típusú adatok kezelésének szokásos módszere
egy *kódoló–dekódoló* architektúra tervezése (:numref:`fig_encoder_decoder`),
amely két fő összetevőből áll:
egy *kódolóból*, amely változó hosszúságú sorozatot vesz bemenetként,
és egy *dekódolóból*, amely feltételes nyelvmodellként működik,
feldolgozza a kódolt bemenetet
és a célsorozat bal oldali kontextusát,
és megjósolja a célsorozat következő tokenjét.


![A kódoló–dekódoló architektúra.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Vegyük példaként az angolról franciára való gépi fordítást.
Adott egy angol bemeneti sorozat:
"They", "are", "watching", ".",
ez a kódoló–dekódoló architektúra
először kódolja a változó hosszúságú bemenetet egy állapottá,
majd az állapotot dekódolja,
hogy tokenenként generálja a lefordított sorozatot:
"Ils", "regardent", ".".
Mivel a kódoló–dekódoló architektúra
alkotja a különböző sorozatból sorozatba irányuló modellek alapját
a következő részekben,
ez a rész ezt az architektúrát egy interfésszé alakítja,
amelyet majd implementálunk.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## (**Kódoló**)

A kódoló interfészben
csupán annyit specifikálunk,
hogy a kódoló változó hosszúságú sorozatokat vesz bemenetként `X` formájában.
Az implementációt bármely, ezt az alap `Encoder` osztályt örökölő modell biztosítja.

```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """Az alap kódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """Az alap kódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """Az alap kódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """Az alap kódoló interfész a kódoló–dekódoló architektúrához."""
    def setup(self):
        raise NotImplementedError

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## **Dekódoló**

A következő dekódoló interfészben
egy kiegészítő `init_state` metódust adunk hozzá,
amely a kódoló kimenetét (`enc_all_outputs`)
kódolt állapottá alakítja.
Vegyük figyelembe, hogy ehhez a lépéshez
extra bemenetek is szükségesek lehetnek,
például a bemenet érvényes hossza,
amelyet a :numref:`sec_machine_translation` fejezetben magyaráztunk el.
Változó hosszúságú sorozat tokenenként való generálásához
a dekódoló minden alkalommal egy bemenetet
(pl. az előző időlépésnél generált tokent)
és a kódolt állapotot leképezheti
az aktuális időlépés egy kimeneti tokenévé.

```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """Az alap dekódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """Az alap dekódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """Az alap dekódoló interfész a kódoló–dekódoló architektúrához."""
    def __init__(self):
        super().__init__()

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """Az alap dekódoló interfész a kódoló–dekódoló architektúrához."""
    def setup(self):
        raise NotImplementedError

    # Később lehetnek további argumentumok (pl. kitöltés nélküli hossz)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## **A Kódoló és a Dekódoló Összeillesztése**

Az előre irányú terjesztésben
a kódoló kimenete a kódolt állapot előállítására szolgál,
és ezt az állapotot a dekódoló az egyik bemenetként fogja tovább használni.

```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """Az alap osztály a kódoló–dekódoló architektúrához."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Csak a dekódoló kimenetét adjuk vissza
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """Az alap osztály a kódoló–dekódoló architektúrához."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Csak a dekódoló kimenetét adjuk vissza
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """Az alap osztály a kódoló–dekódoló architektúrához."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Csak a dekódoló kimenetét adjuk vissza
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

A következő részben látni fogjuk, hogyan alkalmazhatók az RNN-ek
sorozatból sorozatba irányuló modellek tervezéséhez
ezen kódoló–dekódoló architektúra alapján.


## Összefoglalás

A kódoló–dekódoló architektúrák
képesek kezelni olyan bemeneteket és kimeneteket,
amelyek mindketteje változó hosszúságú sorozatokból áll,
és így alkalmasak sorozatból sorozatba irányuló problémákra,
mint amilyen a gépi fordítás.
A kódoló változó hosszúságú sorozatot vesz bemenetként,
és rögzített alakú állapottá alakítja.
A dekódoló a rögzített alakú kódolt állapotot
változó hosszúságú sorozattá képezi le.


## Feladatok

1. Tegyük fel, hogy neurális hálózatokat használunk a kódoló–dekódoló architektúra implementálásához. A kódolónak és a dekódolónak azonos típusú neurális hálózatnak kell lennie?
1. A gépi fordítás mellett gondolj egy másik alkalmazásra, ahol a kódoló–dekódoló architektúra alkalmazható.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18021)
:end_tab:
