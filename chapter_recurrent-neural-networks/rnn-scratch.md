# Rekurrens Neurális Hálózat Implementálása az Alapoktól
:label:`sec_rnn-scratch`

Most készen állunk egy RNN implementálására az alapoktól.
Különösen, ezt az RNN-t karakter szintű nyelvmodellként fogjuk tanítani
(ld. :numref:`sec_rnn`),
és egy H. G. Wells *Az időgép* c. könyvének teljes szövegéből álló
korpuszon tanítjuk,
a :numref:`sec_text-sequence` fejezetben vázolt
adatfeldolgozási lépések szerint.
Az adathalmaz betöltésével kezdünk.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN Modell

Egy osztály definiálásával kezdjük
az RNN modell implementálásához
(:numref:`subsec_rnn_w_hidden_states`).
Megjegyezzük, hogy a rejtett egységek `num_hiddens` száma
egy hangolható hiperparaméter.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """Az alapoktól implementált RNN modell."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """Az alapoktól implementált RNN modell."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**Az alábbi `forward` metódus meghatározza, hogyan kell kiszámítani
a kimenetet és a rejtett állapotot bármely időlépésnél,
az aktuális bemenet és a modell előző időlépésbeli állapota alapján.**]
Megjegyezzük, hogy az RNN modell végighurkol az `inputs`
legkülső dimenzióján,
egy időlépésenként frissítve a rejtett állapotot.
A modell itt $\tanh$ aktivációs függvényt használ (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Kezdeti állapot alakja: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Az inputs alakja: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Az inputs alakja: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Bemeneti sorozatok mini-batch-jét a következőképpen adhatjuk be egy RNN modellbe.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Ellenőrizzük, hogy az RNN modell
helyes alakú eredményeket produkál-e,
biztosítva, hogy a rejtett állapot dimenziójának mérete változatlan marad.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Ellenőrzi egy lista hosszát."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Ellenőrzi egy tenzor alakját."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNN-Alapú Nyelvmodell

A következő `RNNLMScratch` osztály egy
RNN-alapú nyelvmodellt definiál,
ahol az RNN-t az `__init__` metódus `rnn` argumentumán
keresztül adjuk át.
Nyelvmodellek tanításakor
a bemenetek és kimenetek
ugyanabból a szókincsből valók.
Ezért azonos dimenzióval rendelkeznek,
amely egyenlő a szókincs méretével.
Megjegyezzük, hogy perplexitást alkalmazunk a modell értékelésére.
Ahogy a :numref:`subsec_perplexity` fejezetben tárgyaltuk, ez biztosítja
hogy a különböző hosszúságú sorozatok összehasonlíthatók legyenek.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """Az alapoktól implementált RNN-alapú nyelvmodell."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """Az alapoktól implementált RNN-alapú nyelvmodell."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """Az alapoktól implementált RNN-alapú nyelvmodell."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**Egyforró kódolás**]

Emlékeztünk arra, hogy minden tokent
egy numerikus index képviseli, amely jelzi a
megfelelő szó/karakter/szórészlet helyzetét a szókincsben.
Kísértésbe eshetünk, hogy egyetlen bemeneti csomóponttal
(minden időlépésnél) felépítsünk egy neurális hálózatot,
ahol az index skalárértékként adható be.
Ez akkor működik, ha numerikus bemenetekkel, például árral vagy hőmérséklettel dolgozunk,
ahol bármely két kellően közel lévő értéket
hasonlóan kell kezelni.
Ez azonban nem egészen helyes értelmezés.
A szókincsünk $45.$ és $46.$ szavai véletlenül a "their" és "said",
amelyek jelentései egyáltalán nem hasonlóak.

Az ilyen kategorikus adatokkal való munka során
a legelterjedtebb stratégia az, hogy minden elemet
egy *egyforró kódolással* ábrázolunk
(visszagondolva a :numref:`subsec_classification-problem` fejezetből).
Az egyforró kódolás egy olyan vektor, amelynek hosszát
a szókincs $N$ mérete adja meg,
ahol az összes bejegyzés $0$-ra van állítva,
kivéve a tokenünknek megfelelő bejegyzést, amely $1$-re van állítva.
Például, ha a szókincsnek öt eleme lenne,
akkor a 0-s és 2-es indexeknek megfelelő egyforró vektorok a következők lennének.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**Az egyes iterációknál mintavételezett mini-batch-ek
(batch méret, időlépések száma) alakot vesznek fel.
Miután minden bemenetet egyforró vektorként ábrázoltunk,
minden mini-batch-t háromdimenziós tenzorként kezelhetünk,
ahol a harmadik tengely mentén lévő hosszt
a szókincs mérete adja meg (`len(vocab)`).**)
Gyakran transzponáljuk a bemenetet, hogy
(időlépések száma, batch méret, szókincs mérete) alakú kimenetet kapjunk.
Ez lehetővé teszi, hogy kényelmesebben végighurkolhassunk a legkülső dimenzión
egy mini-batch rejtett állapotainak frissítésekor,
időlépésenként
(pl. a fenti `forward` metódusban).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Kimenet alakja: (num_steps, batch_size, vocab_size)
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### RNN kimenetek átalakítása

A nyelvmodell egy teljesen összekötött kimeneti réteget alkalmaz
az RNN kimeneteknek token előrejelzésekké alakításához minden időlépésnél.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Ellenőrizzük [**hogy az előremutató számítás
helyes alakú kimeneteket produkál-e.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradiens vágás**]


Bár már megszokott szemszögből gondolsz a neurális hálózatokra
mint "mély" hálózatokra abban az értelemben, hogy sok réteg
választja el a bemenetet és a kimenetet
még egyetlen időlépésen belül is,
a sorozat hossza bevezet
egy új mélységfogalmat.
Amellett, hogy áthalad a hálózaton
a bemenet-kimenet irányban,
az első időlépés bemenetének
át kell haladnia $T$ réteg láncolatán
az időlépések mentén, hogy
befolyásolja a modell kimenetét
az utolsó időlépésnél.
Visszafelé nézve, minden iterációban,
gradienseket terjesztünk vissza az időn keresztül,
ami egy $\mathcal{O}(T)$ hosszúságú mátrixszorzati láncot eredményez.
Ahogy a :numref:`sec_numerical_stability` fejezetben megemlítettük,
ez numerikus instabilitást okozhat,
ami a gradiensek robbanásához vagy eltűnéséhez vezet,
a súlymátrixok tulajdonságaitól függően.

Az eltűnő és robbanó gradiensek kezelése
alapvető probléma az RNN-ek tervezésekor,
és néhány legnagyobb előrelépés inspirálója lett
a modern neurális hálózat architektúrákban.
A következő fejezetben azokról
a speciális architektúrákról fogunk beszélni, amelyeket
az eltűnő gradiens problémájának enyhítése reményében terveztek.
Azonban még a modern RNN-ek is gyakran szenvednek
a robbanó gradiensektől.
Egy elegánstalan, de mindenütt jelen lévő megoldás
az, hogy egyszerűen *levágjuk* a gradienseket,
arra kényszerítve az így "levágott" gradienseket,
hogy kisebb értékeket vegyenek fel.


Általánosan szólva, amikor valamely célfüggvényt
gradienscsökkenés módszerrel optimalizálunk, iteratívan frissítjük
az érdeklődési paraméterünket, mondjuk egy $\mathbf{x}$ vektort,
de a negatív $\mathbf{g}$ gradiens irányában toljuk
(sztochasztikus gradienscsökkenés esetén
ezt a gradienst
egy véletlenszerűen mintavételezett mini-batch-en számítjuk ki).
Például $\eta > 0$ tanulási rátával
minden frissítés $\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$ formát ölt.
Tegyük fel tovább, hogy az $f$ célfüggvény
kellően sima.
Formálisan azt mondjuk, hogy a cél
*Lipschitz-folytonos* $L$ konstanssal,
ami azt jelenti, hogy bármely $\mathbf{x}$ és $\mathbf{y}$ esetén fennáll:

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Ahogy látható, amikor a paramétervektor frissítésekor kivonjuk $\eta \mathbf{g}$-t,
a cél értékének változása
a tanulási rátától,
a gradiens normájától és $L$-től függ a következőképpen:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

Más szóval, a cél nem változhat
$L \eta \|\mathbf{g}\|$-nél többet.
Ennek a felső korlátnak a kis értéke
jónak vagy rossznak is tekinthető.
A hátrány az, hogy korlátozzuk a sebességet,
amellyel csökkenthetjük a cél értékét.
A jó oldal az, hogy ez korlátozza, mennyire mehetünk félre
bármely egy gradienslépésnél.


Ha azt mondjuk, hogy a gradiensek robbannak,
azt értjük, hogy $\|\mathbf{g}\|$
túlságosan naggyá válik.
Ebben a legrosszabb esetben akkora kárt okozhatnánk
egyetlen gradienslépéssel, hogy visszavonnánk
az összes olyan fejlődést, amelyet
ezernyi tanítási iteráció alatt értünk el.
Amikor a gradiensek ilyen nagyok lehetnek,
a neurális hálózat tanítása gyakran széttart,
nem sikerül csökkenteni a cél értékét.
Más esetekben a tanítás végül konvergál,
de instabil, mivel a veszteségben hatalmas ugrások vannak.


Az $L \eta \|\mathbf{g}\|$ méretének korlátozásának egyik módja
a $\eta$ tanulási ráta csökkentése apró értékekre.
Ennek az az előnye, hogy nem torzítjuk a frissítéseket.
De mi van, ha csak *ritkán* kapunk nagy gradienseket?
Ez a drasztikus lépés lelassítja haladásunkat minden lépésnél,
csak azért, hogy megbirkózzunk a ritka robbanó gradiens eseményekkel.
Egy népszerű alternatíva a *gradiens vágás* heurisztika alkalmazása,
amely vetíti a $\mathbf{g}$ gradienseket egy adott $\theta$ sugarú gömbgömbre
a következőképpen:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Ez biztosítja, hogy a gradiens normája soha ne haladja meg $\theta$-t,
és hogy a frissített gradiens teljes mértékben igazodik
$\mathbf{g}$ eredeti irányához.
Emellett kívánatos mellékhatása is van,
hogy korlátozza, mennyit befolyásolhat
bármely adott mini-batch
(és azon belül bármely adott minta)
a paramétervektort.
Ez bizonyos fokú robusztusságot ad a modellnek.
Hogy egyértelmű legyek: ez egy trükk.
A gradiens vágás azt jelenti, hogy nem mindig
követjük az igazi gradienst, és nehéz
analitikusan érvelni a lehetséges mellékhatásokról.
Azonban nagyon hasznos trükk,
és széles körben alkalmazzák az RNN implementációkban
a legtöbb deep learning keretrendszerben.


Az alábbiakban definiálunk egy metódust a gradiensek levágásához,
amelyet a `d2l.Trainer` osztály `fit_epoch` metódusa hív meg
(ld. :numref:`sec_linear_scratch`).
Megjegyezzük, hogy a gradiens normájának kiszámításakor
összefűzzük az összes modellparamétert,
egyetlen hatalmas paramétervektorként kezelve őket.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Tanítás

*Az időgép* adathalmazt (`data`) felhasználva,
az alapoktól implementált RNN (`rnn`) alapján
egy karakter szintű nyelvmodellt (`model`) tanítunk.
Megjegyezzük, hogy először kiszámítjuk a gradienseket,
majd levágjuk őket, végül pedig
a levágott gradiensek felhasználásával frissítjük a modell paramétereit.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Dekódolás

Miután megtanítottunk egy nyelvmodellt,
nemcsak a következő token megjóslásához használhatjuk,
hanem a következő tokenek folyamatos megjóslásához is,
az előzőleg megjósolt tokent a bemenet következő tagjaként kezelve.
Néha csak szöveget szeretnénk generálni,
mintha egy dokumentum elejétől kezdenénk.
Ugyanakkor hasznos a nyelvmodellt
egy felhasználó által megadott előtagra kondicionálni.
Például ha egy keresőmotor
automatikus kiegészítési funkcióját fejlesztenénk,
vagy segítenénk a felhasználóknak e-mailek írásában,
be kellene táplálnunk azt, amit
eddig írtak (az előtagot),
majd generálnánk egy valószínű folytatást.


[**A következő `predict` metódus
egy karakterenkénti folytatást generál,
miután befogadta a felhasználó által megadott `prefix`-et**].
Az `prefix` karakterein való végighurklásakor
folyamatosan adjuk tovább a rejtett állapotot
a következő időlépésre,
de nem generálunk kimenetet.
Ezt *bemelegítési* periódusnak nevezzük.
Az előtag befogadása után készen állunk
arra, hogy elkezdük kibocsátani a következő karaktereket,
amelyek mindegyike visszatáplálódik a modellbe
a következő időlépés bemenetként.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Bemelegítési periódus
            outputs.append(vocab[prefix[i + 1]])
        else:  # num_preds lépés előrejelzése
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Bemelegítési periódus
            outputs.append(vocab[prefix[i + 1]])
        else:  # num_preds lépés előrejelzése
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

A következőkben meghatározzuk az előtagot,
és generálunk belőle 20 további karaktert.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

Bár a fenti RNN modell alapoktól való implementálása tanulságos, nem kényelmes.
A következő szakaszban meglátjuk, hogyan aknázhatjuk ki a deep learning keretrendszereket az RNN-ek gyors felépítéséhez
standard architektúrák segítségével, és hogyan érhetünk el teljesítménynövekedést
a nagymértékben optimalizált könyvtári függvényekre támaszkodva.


## Összefoglalás

RNN-alapú nyelvmodelleket taníthatunk szöveg generálásához a felhasználó által megadott szöveges előtag alapján.
Egy egyszerű RNN nyelvmodell bemeneti kódolásból, RNN modellezésből és kimenetgenerálásból áll.
Tanítás során a gradiens vágás enyhítheti a robbanó gradiensek problémáját, de nem kezeli az eltűnő gradiensek problémáját. A kísérletben egy egyszerű RNN nyelvmodellt implementáltunk, és karakter szinten tokenizált szöveges sorozatokon tanítottuk gradiens vágással. Egy előtagra kondicionálva egy nyelvmodellt használhatunk valószínű folytatások generálásához, ami számos alkalmazásban, pl. az automatikus kiegészítési funkciókban hasznos.


## Feladatok

1. Az implementált nyelvmodell az *Az időgép* összes múltbeli tokenje alapján jósolja-e meg a következő tokent, egészen az első tokenig visszamenve?
1. Melyik hiperparaméter szabályozza az előrejelzéshez felhasznált előzmény hosszát?
1. Mutasd meg, hogy az egyforró kódolás ekvivalens azzal, hogy minden objektumhoz különböző beágyazást választunk!
1. Állítsd be a hiperparamétereket (pl. epochok száma, rejtett egységek száma, időlépések száma egy mini-batch-ben és a tanulási ráta) a perplexitás javításához! Milyen alacsonyra mehetsz le, ha ragaszkodsz ehhez az egyszerű architektúrához?
1. Cseréld fel az egyforró kódolást tanulható beágyazásokra! Ez jobb teljesítményhez vezet-e?
1. Végezz kísérletet annak meghatározásához, hogy ez az *Az időgépen* tanított nyelvmodell
   milyen jól működik H. G. Wells más könyvein,
   pl. *A világok harcán*.
1. Végezz egy másik kísérletet a modell perplexitásának értékeléséhez
   más szerzők könyvein.
1. Módosítsd az előrejelzési metódust úgy, hogy mintavételezést alkalmazzon
   ahelyett, hogy a legvalószínűbb következő karaktert választaná ki.
    * Mi történik?
    * Tedd elfogulttá a modellt a valószínűbb kimenetek felé, pl.
    mintavételezéssel $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$-ból $\alpha > 1$ esetén.
1. Futtasd a kód ebben a szakaszban a gradiens levágása nélkül! Mi történik?
1. Cseréld fel az ebben a szakaszban alkalmazott aktivációs függvényt ReLU-ra,
   és ismételd meg a kísérleteket ebben a szakaszban. Még mindig szükségünk van gradiens vágásra? Miért?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18014)
:end_tab:
