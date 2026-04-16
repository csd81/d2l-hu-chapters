# RMSProp
:label:`sec_rmsprop`


A :numref:`sec_adagrad` szakasz egyik kulcsproblémája, hogy a tanulási sebesség előre meghatározott $\mathcal{O}(t^{-\frac{1}{2}})$ ütemben csökken. Bár ez általában megfelelő konvex problémáknál, nemkonvex esetekben – mint a mélytanulásban – nem biztos, hogy ideális. Az Adagrad koordinátánkénti alkalmazkodóképessége azonban előkondicionálóként rendkívül kívánatos.

:citet:`Tieleman.Hinton.2012` az RMSProp algoritmust javasolta egyszerű megoldásként, amely szétválasztja az ütemezési sebességet a koordinátánkénti adaptív tanulási sebességektől. A probléma az, hogy az Adagrad a $\mathbf{g}_t$ gradiens négyzetét halmozza fel a $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ állapotvektorba. Ennek eredményeként $\mathbf{s}_t$ normalizáció hiányában határ nélkül nő, lényegében lineárisan az algoritmus konvergálásával.

Ennek a problémának egyik megoldása az $\mathbf{s}_t / t$ alkalmazása. A $\mathbf{g}_t$ ésszerű eloszlásaira ez konvergál. Sajnos nagyon hosszú ideig tarthat, amíg a határviselkedés érdemlegessé válik, mivel az eljárás az értékek teljes trajektóriájára emlékezik. Alternatíva a momentum módszerben alkalmazott kiszivárgó átlag, vagyis $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ valamely $\gamma > 0$ paraméterrel. Az összes többi rész változatlanul hagyása adja az RMSPropot.

## Az algoritmus

Írjuk fel részletesen az egyenleteket:

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Az $\epsilon > 0$ konstanst általában $10^{-6}$-ra állítják, hogy elkerüljük a nullával való osztást vagy a túlságosan nagy lépésméreteket. Ezen kiterjesztés alapján most már szabadon szabályozhatjuk a $\eta$ tanulási sebességet a koordinátánkénti skálázástól függetlenül. A kiszivárgó átlagok tekintetében ugyanazt a logikát alkalmazhatjuk, mint a momentum módszernél. A $\mathbf{s}_t$ definíciójának kifejtésével:

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Ahogy korábban a :numref:`sec_momentum` szakaszban, itt is alkalmazzuk az $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$ összefüggést. Tehát a súlyok összege $1$-re normalizált, és egy megfigyelés felezési ideje $\gamma^{-1}$. Jelenítsük meg a múlt 40 időlépésre vonatkozó súlyokat $\gamma$ különböző értékeire.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Implementálás alapoktól

Mint korábban, az $f(\mathbf{x})=0.1x_1^2+2x_2^2$ másodfokú függvényt alkalmazva megfigyeljük az RMSProp trajektóriáját. Felidézve a :numref:`sec_adagrad` szakaszt: amikor az Adagradet 0.4-es tanulási sebességgel alkalmaztuk, a változók nagyon lassan mozogtak az algoritmus késői szakaszaiban, mert a tanulási sebesség túl gyorsan csökkent. Mivel az RMSPropban $\eta$-t külön szabályozzuk, ez nem fordulhat elő.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Ezután implementáljuk az RMSPropot egy mély hálózatban való alkalmazásra. Ez ugyanolyan egyszerű.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

A kezdeti tanulási sebességet 0.01-re, a $\gamma$ súlyozási paramétert 0.9-re állítjuk. Vagyis $\mathbf{s}$ átlagosan az elmúlt $1/(1-\gamma) = 10$ négyzetes gradiensmegfigyelés felett aggregálódik.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Tömör implementáció

Mivel az RMSProp meglehetősen népszerű algoritmus, a `Trainer` példányban is elérhető. Mindössze annyit kell tennünk, hogy az `rmsprop` névvel inicializáljuk a `gamma1` paraméternek $\gamma$-t adva.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Összefoglalás

* Az RMSProp nagyon hasonlít az Adagradhoz, amennyiben mindkettő a gradiens négyzetét alkalmazza az együtthatók skálázásához.
* Az RMSProp a momentumhoz hasonlóan kiszivárgó átlagolást alkalmaz. Az RMSProp azonban ezt a technikát az együtthatónkénti előkondicionáló módosítására alkalmazza.
* A tanulási sebességet a kísérletezőnek kell a gyakorlatban ütemeznie.
* A $\gamma$ együttható meghatározza, hogy milyen hosszú a korábbi adatok figyelembevételi ablaka a koordinátánkénti skála módosításakor.

## Gyakorló feladatok

1. Mi történik kísérletileg, ha $\gamma = 1$-et állítunk be? Miért?
1. Forgasd el az optimalizálási problémát a $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ minimalizálásához. Mi történik a konvergenciával?
1. Próbáld ki, mi történik az RMSProppal egy valódi gépi tanulási problémán, például Fashion-MNIST tanításakor. Kísérletezz a tanulási sebesség módosításának különböző lehetőségeivel.
1. Módosítanád-e $\gamma$-t az optimalizálás előrehaladtával? Mennyire érzékeny az RMSProp erre?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
