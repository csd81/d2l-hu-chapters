# Generatív adversariális hálózatok
:label:`sec_basic_gan`

A könyv nagy részében arról beszéltünk, hogyan lehet előrejelzéseket készíteni. Valamilyen formában mély neurális hálózatokat alkalmaztunk az adatpéldákból a címkékbe való leképezések megtanulásához. Az ilyen típusú tanulást diszkriminatív tanulásnak nevezzük: pontosan azt szeretnénk megtanulni, hogy megkülönböztessük a macskák és kutyák fotóit egymástól. Az osztályozók és a regresszorok egyaránt a diszkriminatív tanulás példái. A visszaterjesztéssel betanított neurális hálózatok mindent felforgattak, amit a diszkriminatív tanulásról tudni vélünk nagy és bonyolult adathalmazok esetén. A nagy felbontású képek osztályozási pontossága öt-hat év alatt a használhatatlan szintről az emberi szintre emelkedett (némi fenntartással). Megkímélünk benneteket attól, hogy újra felsoroljuk az összes olyan diszkriminatív feladatot, amelyeken a mély neurális hálózatok lenyűgözően teljesítenek.

De a gépi tanulásban ennél több is van, mint pusztán diszkriminatív feladatok megoldása. Például adott egy nagy adathalmaz, bármilyen címke nélkül: lehet, hogy meg szeretnénk tanulni egy olyan modellt, amely tömören leírja az adatok jellemzőit. Ilyen modell megléte esetén szintetikus adatpéldákat tudnánk mintázni, amelyek hasonlítanak a tanítóadatok eloszlásához. Például adott egy nagy arc-fotógyűjtemény, esetleg szeretnénk tudni előállítani egy új, fotórealisztikus képet, amely úgy néz ki, mintha valószínűleg ugyanabból az adathalmazból származna. Az ilyen típusú tanulást generatív modellezésnek hívják.

Egészen a közelmúltig nem volt módszerünk fotórealisztikus új képek szintézisére. A mély neurális hálózatok diszkriminatív tanulásban elért sikere azonban új lehetőségeket nyitott meg. Az elmúlt három év egyik nagy trendje a diszkriminatív mély hálózatok alkalmazása azoknak a kihívásoknak a leküzdésére, amelyeket általában nem felügyelt tanulási problémáknak tekintünk. A rekurrens neurális hálózat nyelvi modellek egypéldái a diszkriminatív hálózat (a következő karakter előrejelzésére betanított) felhasználásának, amely betanítás után generatív modellként is működhet.

2014-ben egy áttörést hozó cikk bemutatta a generatív adversariális hálózatokat (GAN-ok) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`, a diszkriminatív modellek erejének kiaknázására kitalált új, szellemes módszert, amellyel jó generatív modelleket lehet előállítani. A GAN-ok lényege az az ötlet, hogy egy adatgenerátor akkor jó, ha nem tudjuk megkülönböztetni a hamis adatokat a valódi adatoktól. A statisztikában ezt kétmintás tesztnek hívják — egy olyan tesztnek, amellyel megválaszolhatjuk, hogy a $X=\{x_1,\ldots, x_n\}$ és $X'=\{x'_1,\ldots, x'_n\}$ adathalmazok ugyanabból az eloszlásból lettek-e véve. A legtöbb statisztikai cikk és a GAN-ok közötti fő különbség az, hogy az utóbbiak ezt az ötletet konstruktív módon alkalmazzák. Más szóval: ahelyett, hogy csupán egy olyan modellt tanítanának, amely azt mondja: „hé, ez a két adathalmaz nem úgy néz ki, mint ha ugyanabból az eloszlásból származna", a [kétmintás tesztet](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) tanítási jelként használják egy generatív modellhez. Ez lehetővé teszi, hogy javítsuk az adatgenerátort, amíg olyasmit nem generál, ami hasonlít a valódi adatokhoz. Legalábbis be kell csapnia az osztályozót, még akkor is, ha az osztályozónk egy csúcstechnológiás mély neurális hálózat.

![Generatív adversariális hálózatok](../img/gan.svg)
:label:`fig_gan`


A GAN architektúrát a :numref:`fig_gan` ábra szemlélteti.
Amint láthatod, a GAN architektúrában két rész van: először szükségünk van egy eszközre (mondjuk egy mély hálózatra, de valójában bármi lehet, például egy játékrendező motor), amely potenciálisan képes olyan adatokat generálni, amelyek megkülönböztethetetlenek a valóditól. Képekkel való foglalkozás esetén ennek képeket kell generálnia. Ha hanggal foglalkozunk, hangsorozatokat kell generálnia, és így tovább. Ezt nevezzük generátor hálózatnak. A második összetevő a diszkriminátor hálózat. Ez megpróbálja megkülönböztetni a hamis és a valódi adatokat egymástól. Mindkét hálózat versenyez egymással. A generátor hálózat megpróbálja becsapni a diszkriminátor hálózatot. Eközben a diszkriminátor hálózat alkalmazkodik az új hamis adatokhoz. Ez az információ viszont a generátor hálózat fejlesztésére szolgál, és így tovább.


A diszkriminátor egy bináris osztályozó, amely megkülönbözteti, hogy az $x$ bemeneti adat valódi-e (valós adatból) vagy hamis (a generátorból). Általában a diszkriminátor egy skaláris $o\in\mathbb R$ előrejelzést ad ki az $\mathbf x$ bemenetre, például egy rejtett méretű 1-es teljesen összekötött réteggel, majd szigmoid függvényt alkalmaz a becsült $D(\mathbf x) = 1/(1+e^{-o})$ valószínűség előállítására. Feltételezzük, hogy a valódi adatok $y$ címkéje $1$, a hamis adatoké $0$. A diszkriminátort a kereszt-entrópia veszteség minimalizálására tanítjuk, azaz:

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

A generátor először egy $\mathbf z\in\mathbb R^d$ paramétert mintáz egy véletlen forrásból, például egy normális eloszlásból: $\mathbf z \sim \mathcal{N} (0, 1)$. A $\mathbf z$ értéket látens változónak nevezzük.
Ezután egy függvényt alkalmaz a $\mathbf x'=G(\mathbf z)$ generálásához. A generátor célja, hogy becsapja a diszkriminátort, hogy az $\mathbf x'=G(\mathbf z)$ értéket valódi adatnak minősítse, azaz $D( G(\mathbf z)) \approx 1$ legyen.
Más szóval, adott $D$ diszkriminátor esetén frissítjük a $G$ generátor paramétereit a kereszt-entrópia veszteség maximalizálására, amikor $y=0$, azaz:

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

Ha a generátor tökéletes munkát végez, akkor $D(\mathbf x')\approx 1$, és így a fenti veszteség közel nulla, ami túl kis gradienseket eredményez ahhoz, hogy a diszkriminátor jó haladást érjen el. Ezért általánosan a következő veszteséget minimalizáljuk:

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

ami csak azt jelenti, hogy az $\mathbf x'=G(\mathbf z)$ értéket betápláljuk a diszkriminátorba, de $y=1$ címkét adunk.


Összefoglalva, $D$ és $G$ egy „minimax" játékot játszik az alábbi átfogó célfüggvénnyel:

$$\min_D \max_G \{ -E_{x \sim \textrm{Data}} \log D(\mathbf x) - E_{z \sim \textrm{Noise}} \log(1 - D(G(\mathbf z))) \}.$$



A GAN-ok alkalmazásainak nagy része képekkel kapcsolatos. Demonstrációs célból először egy sokkal egyszerűbb eloszlás illesztésével elégedszünk meg. Bemutatjuk, mi történik, ha GAN-okat használunk a világ leghatékonytalanabb Gauss-paraméter-becslőjének megépítéséhez. Kezdjük is el.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Néhány „valódi" adat generálása

Mivel ez a világ legsablonosabb példája lesz, egyszerűen Gauss-eloszlásból vett adatokat generálunk.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

Nézzük meg, mit kaptunk. Ennek egy valahogy tetszőlegesen eltolt Gauss-eloszlásnak kell lennie, amelynek átlaga $b$ és kovariancia-mátrixa $A^TA$.

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## Generátor

A generátor hálózatunk a lehető legegyszerűbb hálózat lesz — egyetlen rétegű lineáris modell. Ez azért van, mert ezt a lineáris hálózatot Gauss-adatgenerátorral fogjuk hajtani. Ezért szó szerint csak a paramétereket kell megtanulnia a tökéletes hamisításhoz.

```{.python .input}
#@tab mxnet
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## Diszkriminátor

A diszkriminátor esetén egy kicsit válogatósabbak leszünk: 3 rétegből álló MLP-t fogunk használni, hogy kissé érdekesebb legyen.

```{.python .input}
#@tab mxnet
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## Tanítás

Először definiálunk egy függvényt a diszkriminátor frissítéséhez.

```{.python .input}
#@tab mxnet
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Frissíti a diszkriminátort."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Nem kell gradienseket számítani a `net_G`-hez, leválasztjuk a
        # gradiens-számításból.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Frissíti a diszkriminátort."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Nem kell gradienseket számítani a `net_G`-hez, leválasztjuk a
    # gradiens-számításból.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """Frissíti a diszkriminátort."""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # Valódi adatokhoz tartozó címkék
    zeros = tf.zeros((batch_size,)) # Hamis adatokhoz tartozó címkék
    # Nem kell gradienseket számítani a `net_G`-hez, így kívül van a GradientTape-en
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # A veszteséget megszorozzuk batch_size-zal, hogy megfeleljen a PyTorch BCEWithLogitsLoss-ának
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

A generátort hasonlóan frissítjük. Újra felhasználjuk a kereszt-entrópia veszteséget, de a hamis adatok $0$ értékű címkéjét $1$-re változtatjuk.

```{.python .input}
#@tab mxnet
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Frissíti a generátort."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # Az `update_D`-ből újra felhasználhatnánk a `fake_X`-et a számítás megtakarításához
        fake_X = net_G(Z)
        # A `fake_Y` újraszámítása szükséges, mivel `net_D` megváltozott
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Frissíti a generátort."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # Az `update_D`-ből újra felhasználhatnánk a `fake_X`-et a számítás megtakarításához
    fake_X = net_G(Z)
    # A `fake_Y` újraszámítása szükséges, mivel `net_D` megváltozott
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """Frissíti a generátort."""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # Az `update_D`-ből újra felhasználhatnánk a `fake_X`-et a számítás megtakarításához
        fake_X = net_G(Z)
        # A `fake_Y` újraszámítása szükséges, mivel `net_D` megváltozott
        fake_Y = net_D(fake_X)
        # A veszteséget megszorozzuk batch_size-zal, hogy megfeleljen a PyTorch BCEWithLogits loss-ának
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

Mind a diszkriminátor, mind a generátor bináris logisztikus regressziót hajt végre kereszt-entrópia veszteséggel. A tanítási folyamat simábbá tételéhez Adamot használunk. Minden iterációban először a diszkriminátort, majd a generátort frissítjük. Mindkét veszteséget és a generált példákat megjelenítjük.

```{.python .input}
#@tab mxnet
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Egy epoch tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # A generált példák megjelenítése
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Egy epoch tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # A generált példák megjelenítése
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Egy epoch tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # A generált példák megjelenítése
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])

        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))

    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

Most megadjuk a hiperparamétereket a Gauss-eloszlás illesztéséhez.

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## Összefoglalás

* A generatív adversariális hálózatok (GAN-ok) két mély hálózatból állnak: a generátorból és a diszkriminátorból.
* A generátor a valódi képhez minél közelebb álló képet generál, hogy becsapja a diszkriminátort, a kereszt-entrópia veszteség maximalizálásával, azaz $\max \log(D(\mathbf{x'}))$.
* A diszkriminátor megpróbálja megkülönböztetni a generált képeket a valódi képektől, a kereszt-entrópia veszteség minimalizálásával, azaz $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$.

## Feladatok

* Létezik-e egyensúly, amelyben a generátor nyer, azaz a diszkriminátor végül képtelen megkülönböztetni a két eloszlást véges mintákon?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/408)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1082)
:end_tab:
