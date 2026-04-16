# Adam
:label:`sec_adam`

Az ebbe a szakaszba vezető tárgyalások során számos technikával találkoztunk a hatékony optimalizálás érdekében. Összefoglaljuk ezeket részletesen:

* Láttuk, hogy a :numref:`sec_sgd` szakasz hatékonyabb, mint a Gradient Descent optimalizálási problémák megoldásánál, pl. a redundáns adatokkal szembeni belső rugalmassága miatt.
* Láttuk, hogy a :numref:`sec_minibatch_sgd` szakasz a vektorizálásból eredő jelentős további hatékonyságot kínál, az egyik minibatchben nagyobb megfigyeléssorozatokat alkalmazva. Ez a kulcsa a hatékony multi-gépes, multi-GPU és összességében párhuzamos feldolgozásnak.
* A :numref:`sec_momentum` szakasz hozzáadott egy mechanizmust a korábbi gradiensek előzményeinek összesítésére a konvergencia gyorsítása érdekében.
* A :numref:`sec_adagrad` szakasz koordinátánkénti skálázást alkalmazott, hogy lehetővé tegye a számításilag hatékony előkondicionálást.
* A :numref:`sec_rmsprop` szakasz szétválasztotta a koordinátánkénti skálázást a tanulási sebesség módosításától.

Az Adam :cite:`Kingma.Ba.2014` mindezeket a technikákat egyetlen hatékony tanulási algoritmusba foglalja. Ahogy várható, ez lett az egyik legnépszerűbb és legrobusztusabb optimalizálási algoritmus a mélytanulásban. Nem mentes a problémáktól sem: különösen :cite:`Reddi.Kale.Kumar.2019` megmutatja, hogy vannak olyan helyzetek, amikor az Adam divergálhat a rossz varianciaszabályozás miatt. Egy rákövetkező munkában :citet:`Zaheer.Reddi.Sachan.ea.2018` az Adam egy javított változatát javasolta Yogi néven, amely ezeket a problémákat kezeli. Erről bővebben később. Egyelőre tekintsük át az Adam algoritmust.

## Az algoritmus

Az Adam egyik kulcseleme, hogy exponenciálisan súlyozott mozgóátlagokat (más névvel kiszivárgó átlagokat) alkalmaz a momentum és a gradiens második momentumának becslésére. Vagyis a következő állapotváltozókat alkalmazza:

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

ahol $\beta_1$ és $\beta_2$ nemnegatív súlyozási paraméterek. Általánosan használt értékeik $\beta_1 = 0.9$ és $\beta_2 = 0.999$. Vagyis a varianciabecslés *sokkal lassabban* változik, mint a momentum tag. Fontos megjegyezni, hogy ha $\mathbf{v}_0 = \mathbf{s}_0 = 0$-val inicializálunk, kezdetben jelentős torzítás mutatkozik a kisebb értékek felé. Ez kezelhető azzal, hogy $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ felhasználásával renormalizálunk. Ennek megfelelően a normalizált állapotváltozók:

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

A megfelelő becslések alapján most már felírhatjuk a frissítési egyenleteket. Először az RMSProphoz nagyon hasonló módon átskálázzuk a gradienst:

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Az RMSProppal ellentétben a frissítésünk a $\hat{\mathbf{v}}_t$ momentumot alkalmazza, nem magát a gradienst. Ráadásul enyhe esztétikai különbség is van: az átskálázás $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$-t alkalmaz $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ helyett. Az előbbi a gyakorlatban kissé jobbnak bizonyul, ezért tér el az RMSPropptól. Általában $\epsilon = 10^{-6}$-ot választunk a numerikus stabilitás és hűség közötti jó kompromisszumként.

Most már minden összetevő rendelkezésre áll a frissítések kiszámításához. Ez kissé ünneprontó, és egy egyszerű frissítési képletet kapunk:

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Az Adam tervezését áttekintve az ihlete egyértelmű. A momentum és a skálázás egyértelműen látható az állapotváltozókban. Furcsa definíciójuk arra kényszerít minket, hogy korrigáljuk a torzítást (ez kissé eltérő inicializálással és frissítési feltétellel is megoldható lenne). Másodszor, mindkét tag kombinálása meglehetősen egyszerű az RMSProp alapján. Végül az explicit $\eta$ tanulási sebesség lehetővé teszi a lépéshossz szabályozását a konvergencia problémáinak kezelésére.

## Implementáció

Az Adam alapoktól való implementálása nem különösebben ijesztő. Kényelmi okokból a $t$ időlépés számlálót a `hyperparams` szótárban tároljuk. A többi egyszerű.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Készen állunk az Adam alkalmazására a modell tanításához. $\eta = 0.01$ tanulási sebességet alkalmazunk.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Tömörebb implementáció egyszerűen megvalósítható, mivel az `adam` egyike a Gluon `trainer` optimalizálási könyvtárának részeként biztosított algoritmusoknak. Ezért a Gluon implementációhoz csupán konfigurációs paramétereket kell átadni.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Az Adam egyik problémája, hogy még konvex esetekben is elveszítheti a konvergenciát, ha a $\mathbf{s}_t$-beli második momentum becslése túl nagyra nő. Megoldásként :citet:`Zaheer.Reddi.Sachan.ea.2018` egy finomított frissítést (és inicializálást) javasolt $\mathbf{s}_t$-re. A folyamat megértéséhez írjuk át az Adam frissítését a következőképpen:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Ha $\mathbf{g}_t^2$ nagy varianciájú, vagy a frissítések ritkák, $\mathbf{s}_t$ túl gyorsan felejtheti a múltbeli értékeket. Lehetséges megoldás a $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ helyettesítése $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$-vel. Ekkor a frissítés nagysága nem függ az eltérés mértékétől. Ez adja a Yogi frissítéseket:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

A szerzők ezen kívül javasolják, hogy a momentumot nagyobb kezdeti batch alapján inicializáljuk, nem csupán a kezdeti pontszerű becslésből. Az implementáció részleteit elhagyjuk, mivel nem lényegesek a tárgyalás szempontjából, és enélkül is jól konvergál.

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Összefoglalás

* Az Adam számos optimalizálási algoritmus jellemzőit kombinálja egy meglehetősen robusztus frissítési szabállyá.
* Az RMSProp alapján létrehozva az Adam EWMA-t is alkalmaz a minibatch sztochasztikus gradiensre.
* Az Adam torzításkorrekciót alkalmaz a lassú indulás kompenzálásához a momentum és a második momentum becslésekor.
* Jelentős varianciával rendelkező gradiensek esetén konvergenciaproblémák merülhetnek fel. Ezek kezelhetők nagyobb minibatch-ek alkalmazásával, vagy az $\mathbf{s}_t$ javított becslésére való áttéréssel. A Yogi ilyen alternatívát kínál.

## Gyakorló feladatok

1. Módosítsd a tanulási sebességet, és figyeld meg és elemezd a kísérleti eredményeket!
1. Átírhatod-e a momentum és a második momentum frissítéseit úgy, hogy ne igényelje a torzításkorrekciót?
1. Miért kell csökkenteni a $\eta$ tanulási sebességet, ahogy konvergálunk?
1. Próbálj meg olyan esetet konstruálni, amelyben az Adam divergál, de a Yogi konvergál!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
