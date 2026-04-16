# Adadelta
:label:`sec_adadelta`

Az Adadelta az AdaGrad (:numref:`sec_adagrad`) még egy változata. A fő különbség abban rejlik, hogy csökkenti azt a mértéket, amennyire a tanulási ráta a koordinátákhoz alkalmazkodik. Ráadásul hagyományosan úgy emlegetik, mint ami nem rendelkezik tanulási rátával, mivel a változás mértékét magát alkalmazza a jövőbeli változás kalibrálásához. Az algoritmust :citet:`Zeiler.2012` javasolta. Az előző algoritmusok eddigi tárgyalása alapján meglehetősen egyszerűen érthető.

## Az algoritmus

Röviden: az Adadelta két állapotváltozót alkalmaz: $\mathbf{s}_t$-t a gradiens második momentumának kiszivárgó átlagának tárolására, és $\Delta\mathbf{x}_t$-t a modell paramétereiben bekövetkező változás második momentumának kiszivárgó átlagának tárolására. Megjegyezzük, hogy kompatibilitás érdekében a szerzők eredeti jelöléseit és elnevezéseit alkalmazzuk más publikációkkal és implementációkkal (nincs más valódi ok arra, hogy különböző görög változókat használjunk azonos célú paraméterek jelölésére a momentum, Adagrad, RMSProp és Adadelta esetén).

A $\rho$ aktuális paraméterrel az Adadelta technikai részletei a következők. Az alábbi kiszivárgó frissítéseket kapjuk, hasonlóan a :numref:`sec_rmsprop` szakaszhoz:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

A :numref:`sec_rmsprop` szakasztól való különbség az, hogy az átskálázott $\mathbf{g}_t'$ gradienssel végzünk frissítéseket:

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Hogyan számítható az átskálázott $\mathbf{g}_t'$ gradiens? A következőképpen:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

ahol $\Delta \mathbf{x}_{t-1}$ az átskálázott $\mathbf{g}_t'$ gradiensek négyzetének kiszivárgó átlaga. A $\Delta \mathbf{x}_{0}$ értékét $0$-val inicializáljuk, és minden lépésben $\mathbf{g}_t'$-vel frissítjük:

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

és az $\epsilon$ (egy kis érték, például $10^{-5}$) numerikus stabilitás fenntartása érdekében kerül hozzáadásra.



## Implementáció

Az Adadelta minden változóhoz két állapotváltozót igényel: $\mathbf{s}_t$-t és $\Delta\mathbf{x}_t$-t. Ez a következő implementációt eredményezi.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # Helyben végzett frissítések a [:] segítségével
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # Helyben végzett frissítések a [:] segítségével
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

A $\rho = 0.9$ megválasztása minden paraméterfrissítésnél 10-es felezési időt jelent. Ez általában meglehetősen jól működik. Az alábbi viselkedést kapjuk.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

A tömör implementációhoz egyszerűen alkalmazzuk az Adadelta algoritmust a magas szintű API-kból. Ez egy sokkal tömörebb hívást eredményez egyetlen sorban.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# az adadelta az alapértelmezett tanulási rátanél nem konvergál
# de lr = 5.0 esetén konvergál
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Összefoglalás

* Az Adadeltának nincs tanulási ráta paramétere. Ehelyett a paraméterek változásának mértékét alkalmazza a tanulási ráta adaptálásához.
* Az Adadelta két állapotváltozót igényel a gradiens és a paraméterváltozás második momentumainak tárolásához.
* Az Adadelta kiszivárgó átlagokat alkalmaz a megfelelő statisztikák folyamatos becslésének fenntartásához.

## Gyakorló feladatok

1. Módosítsd a $\rho$ értékét. Mi történik?
1. Mutasd meg, hogyan implementálható az algoritmus $\mathbf{g}_t'$ nélkül. Miért lehet ez jó ötlet?
1. Valóban tanulási ráta-mentes-e az Adadelta? Találnál-e olyan optimalizálási problémákat, amelyek megtörik az Adadeltát?
1. Hasonlítsd össze az Adadeltát az Adagraddal és az RMSProppal a konvergencia-viselkedésük tárgyalása érdekében.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
