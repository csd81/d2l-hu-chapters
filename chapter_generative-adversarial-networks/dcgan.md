# Mélyen konvolúciós generatív adversariális hálózatok
:label:`sec_dcgan`

A :numref:`sec_basic_gan` szakaszban bemutattuk a GAN-ok működésének alapötleteit. Megmutattuk, hogy képesek mintákat venni valamilyen egyszerű, könnyen mintázható eloszlásból, például egyenletes vagy normális eloszlásból, és ezeket átalakítani olyan mintákká, amelyek látszólag egyeznek valamely adathalmaz eloszlásával. Bár a 2D Gauss-eloszlás illesztésére vonatkozó példánk elérte a célját, nem volt különösebben izgalmas.

Ebben a szakaszban bemutatjuk, hogyan lehet GAN-okat fotórealisztikus képek generálására használni. Modelleinket a :citet:`Radford.Metz.Chintala.2015` által bevezetett mélyen konvolúciós GAN-okra (DCGAN) alapozzuk. A diszkriminatív számítógépes látás feladatokban bizonyítottan sikeres konvolúciós architektúrát kölcsönözzük, és bemutatjuk, hogyan lehet GAN-ok segítségével fotórealisztikus képek generálására felhasználni.

```{.python .input}
#@tab mxnet
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## A Pokémon adathalmaz

A felhasznált adathalmaz a [pokemondb](https://pokemondb.net/sprites) oldalról kapott Pokémon spritek gyűjteménye. Először töltsük le, csomagoljuk ki és töltsük be az adathalmazt.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = gluon.data.vision.datasets.ImageFolderDataset(data_dir)
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
batch_size = 256
pokemon = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, batch_size=batch_size, image_size=(64, 64))
```

Minden képet $64\times 64$ méretre méretezünk át. A `ToTensor` transzformáció a pixelértékeket $[0, 1]$-be vetíti, míg a generátorunk tanh függvényt alkalmaz a $[-1, 1]$ tartományú kimenetekhez. Ezért az adatokat $0.5$ átlaggal és $0.5$ szórással normalizáljuk az értéktartomány egyeztetéséhez.

```{.python .input}
#@tab mxnet
batch_size = 256
transformer = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(64),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(0.5, 0.5)
])
data_iter = gluon.data.DataLoader(
    pokemon.transform_first(transformer), batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
def transform_func(X):
    X = X / 255.
    X = (X - 0.5) / (0.5)
    return X

# TF>=2.4 esetén használd: `num_parallel_calls = tf.data.AUTOTUNE`
data_iter = pokemon.map(lambda x, y: (transform_func(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
data_iter = data_iter.cache().shuffle(buffer_size=1000).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
```

Jelenítsük meg az első 20 képet.

```{.python .input}
#@tab mxnet
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[:20,:,:,:].transpose(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab pytorch
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize(figsize=(4, 4))
for X, y in data_iter.take(1):
    imgs = X[:20, :, :, :] / 2 + 0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
```

## A generátor

A generátornak az $\mathbf z\in\mathbb R^d$ zajváltozót, egy $d$ hosszúságú vektort, egy $64\times 64$ szélességű és magasságú RGB képpé kell leképeznie. A :numref:`sec_fcn` szakaszban bemutattuk a teljesen konvolúciós hálózatot, amely transzponált konvolúciós rétegeket (lásd :numref:`sec_transposed_conv`) alkalmaz a bemeneti méret növelésére. A generátor alap blokkja egy transzponált konvolúciós rétegből áll, amelyet batchnormalizáció és ReLU aktiváció követ.

```{.python .input}
#@tab mxnet
class G_block(nn.Block):
    def __init__(self, channels, kernel_size=4,
                 strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.Conv2DTranspose(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.Activation('relu')

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input}
#@tab pytorch
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input}
#@tab tensorflow
class G_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 **kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(
            out_channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

Alapértelmezés szerint a transzponált konvolúciós réteg $k_h = k_w = 4$ kernelt, $s_h = s_w = 2$ lépésközt és $p_h = p_w = 1$ párnázást alkalmaz. $n_h^{'} \times n_w^{'} = 16 \times 16$ bemeneti alakzat esetén a generátor blokk megduplázza a bemenet szélességét és magasságát.

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= [(n_h k_h - (n_h-1)(k_h-s_h)- 2p_h] \times [(n_w k_w - (n_w-1)(k_w-s_w)- 2p_w]\\
  &= [(k_h + s_h (n_h-1)- 2p_h] \times [(k_w + s_w (n_w-1)- 2p_w]\\
  &= [(4 + 2 \times (16-1)- 2 \times 1] \times [(4 + 2 \times (16-1)- 2 \times 1]\\
  &= 32 \times 32 .\\
\end{aligned}
$$

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))  # Csatorna-utolsó konvenció
g_blk = G_block(20)
g_blk(x).shape
```

Ha a transzponált konvolúciós réteget $4\times 4$ kernelre, $1\times 1$ lépésközre és nulla párnázásra változtatjuk, akkor $1 \times 1$ bemeneti méret esetén a kimenet szélességét és magasságát 3-mal növeli.

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 1, 1, 3))
# `padding="valid"` megfelel a párnázás nélküli esetnek
g_blk = G_block(20, strides=1, padding="valid")
g_blk(x).shape
```

A generátor négy alap blokkból áll, amelyek a bemenet szélességét és magasságát 1-ről 32-re növelik. Eközben először a látens változót $64\times 8$ csatornára vetíti, majd minden alkalommal felezi a csatornák számát. Végül egy transzponált konvolúciós réteggel generálja a kimenetet. Ez tovább duplázza a szélességet és magasságot, hogy elérje a kívánt $64\times 64$ alakzatot, és a csatornák számát $3$-ra csökkenti. A tanh aktiváló függvény a kimeneti értékeket a $(-1, 1)$ tartományba vetíti.

```{.python .input}
#@tab mxnet
n_G = 64
net_G = nn.Sequential()
net_G.add(G_block(n_G*8, strides=1, padding=0),  # Output: (64 * 8, 4, 4)
          G_block(n_G*4),  # Output: (64 * 4, 8, 8)
          G_block(n_G*2),  # Output: (64 * 2, 16, 16)
          G_block(n_G),    # Output: (64, 32, 32)
          nn.Conv2DTranspose(
              3, kernel_size=4, strides=2, padding=1, use_bias=False,
              activation='tanh'))  # Output: (3, 64, 64)
```

```{.python .input}
#@tab pytorch
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8,
            strides=1, padding=0),                  # Output: (64 * 8, 4, 4)
    G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 8, 8)
    G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 16, 16)
    G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())  # Output: (3, 64, 64)
```

```{.python .input}
#@tab tensorflow
n_G = 64
net_G = tf.keras.Sequential([
    # Output: (4, 4, 64 * 8)
    G_block(out_channels=n_G*8, strides=1, padding="valid"),
    G_block(out_channels=n_G*4), # Output: (8, 8, 64 * 4)
    G_block(out_channels=n_G*2), # Output: (16, 16, 64 * 2)
    G_block(out_channels=n_G), # Output: (32, 32, 64)
    # Output: (64, 64, 3)
    tf.keras.layers.Conv2DTranspose(
        3, kernel_size=4, strides=2, padding="same", use_bias=False,
        activation="tanh")
])
```

Generáljunk egy 100 dimenziós látens változót a generátor kimeneti alakzatának ellenőrzéséhez.

```{.python .input}
#@tab mxnet
x = np.zeros((1, 100, 1, 1))
net_G.initialize()
net_G(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 100, 1, 1))
net_G(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 1, 1, 100))
net_G(x).shape
```

## Diszkriminátor

A diszkriminátor egy normális konvolúciós hálózat, azzal a különbséggel, hogy aktiváló függvényként szivárgó ReLU-t (leaky ReLU) alkalmaz. Adott $\alpha \in[0, 1]$ esetén a definíciója:

$$\textrm{leaky ReLU}(x) = \begin{cases}x & \textrm{if}\ x > 0\\ \alpha x &\textrm{otherwise}\end{cases}.$$

Amint látható, $\alpha=0$ esetén normál ReLU, $\alpha=1$ esetén identitásfüggvény. $\alpha \in (0, 1)$ esetén a szivárgó ReLU egy nemlineáris függvény, amely nem nulla kimenetet ad negatív bemenet esetén. Célja a „haldokló ReLU" probléma megoldása, amikor egy neuron mindig negatív értéket ad ki, és ezért nem tud haladást elérni, mivel a ReLU gradiense 0.

```{.python .input}
#@tab mxnet,pytorch
alphas = [0, .2, .4, .6, .8, 1]
x = d2l.arange(-2, 1, 0.1)
Y = [d2l.numpy(nn.LeakyReLU(alpha)(x)) for alpha in alphas]
d2l.plot(d2l.numpy(x), Y, 'x', 'y', alphas)
```

```{.python .input}
#@tab tensorflow
alphas = [0, .2, .4, .6, .8, 1]
x = tf.range(-2, 1, 0.1)
Y = [tf.keras.layers.LeakyReLU(alpha)(x).numpy() for alpha in alphas]
d2l.plot(x.numpy(), Y, 'x', 'y', alphas)
```

A diszkriminátor alap blokkja egy konvolúciós rétegből áll, amelyet batchnormalizációs réteg és szivárgó ReLU aktiváció követ. A konvolúciós réteg hiperparaméterei hasonlóak a generátor blokk transzponált konvolúciós rétegéhez.

```{.python .input}
#@tab mxnet
class D_block(nn.Block):
    def __init__(self, channels, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2D(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input}
#@tab pytorch
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input}
#@tab tensorflow
class D_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, kernel_size,
                                             strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha)

    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

Az alapértelmezett beállításokkal rendelkező alap blokk felezi a bemenetek szélességét és magasságát, ahogyan azt a :numref:`sec_padding` szakaszban bemutattuk. Például adott $n_h = n_w = 16$ bemeneti alakzat, $k_h = k_w = 4$ kernel alakzat, $s_h = s_w = 2$ lépésköz alakzat és $p_h = p_w = 1$ párnázás alakzat esetén a kimeneti alakzat:

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= \lfloor(n_h-k_h+2p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+2p_w+s_w)/s_w\rfloor\\
  &= \lfloor(16-4+2\times 1+2)/2\rfloor \times \lfloor(16-4+2\times 1+2)/2\rfloor\\
  &= 8 \times 8 .\\
\end{aligned}
$$

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk.initialize()
d_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))
d_blk = D_block(20)
d_blk(x).shape
```

A diszkriminátor a generátor tükörképe.

```{.python .input}
#@tab mxnet
n_D = 64
net_D = nn.Sequential()
net_D.add(D_block(n_D),   # Output: (64, 32, 32)
          D_block(n_D*2),  # Output: (64 * 2, 16, 16)
          D_block(n_D*4),  # Output: (64 * 4, 8, 8)
          D_block(n_D*8),  # Output: (64 * 8, 4, 4)
          nn.Conv2D(1, kernel_size=4, use_bias=False))  # Output: (1, 1, 1)
```

```{.python .input}
#@tab pytorch
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),  # Output: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D*2),  # Output: (64 * 2, 16, 16)
    D_block(in_channels=n_D*2, out_channels=n_D*4),  # Output: (64 * 4, 8, 8)
    D_block(in_channels=n_D*4, out_channels=n_D*8),  # Output: (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))  # Output: (1, 1, 1)
```

```{.python .input}
#@tab tensorflow
n_D = 64
net_D = tf.keras.Sequential([
    D_block(n_D), # Output: (32, 32, 64)
    D_block(out_channels=n_D*2), # Output: (16, 16, 64 * 2)
    D_block(out_channels=n_D*4), # Output: (8, 8, 64 * 4)
    D_block(out_channels=n_D*8), # Outupt: (4, 4, 64 * 64)
    # Output: (1, 1, 1)
    tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
])
```

Egyetlen predikciós értéket kap úgy, hogy utolsó rétegként egy $1$ kimeneti csatornájú konvolúciós réteget alkalmaz.

```{.python .input}
#@tab mxnet
x = np.zeros((1, 3, 64, 64))
net_D.initialize()
net_D(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 3, 64, 64))
net_D(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 64, 64, 3))
net_D(x).shape
```

## Tanítás

A :numref:`sec_basic_gan` alapvető GAN-hoz képest ugyanolyan tanulási rátát alkalmazunk a generátorhoz és a diszkriminátorhoz, mivel hasonlóak egymáshoz. Ezenkívül az Adam (:numref:`sec_adam`) $\beta_1$ paraméterét $0.9$-ről $0.5$-re változtatjuk. Ez csökkenti a momentum simítását — a múltbeli gradiensek exponenciálisan súlyozott mozgóátlagát —, hogy kezelje a gyorsan változó gradienseket, amelyek abból adódnak, hogy a generátor és a diszkriminátor versenyez egymással. Ezenkívül a véletlenszerűen generált `Z` zaj egy 4-dimenziós tenzor, és GPU-t használunk a számítás gyorsítására.

```{.python .input}
#@tab mxnet
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    trainer_hp = {'learning_rate': lr, 'beta1': 0.5}
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', trainer_hp)
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Egy korszak tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.as_in_ctx(device), Z.as_in_ctx(device),
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Generált példák megjelenítése
        Z = np.random.normal(0, 1, size=(21, latent_dim, 1, 1), ctx=device)
        # A szintetikus adatok normalizálása N(0, 1)-re
        fake_x = net_G(Z).transpose(0, 2, 3, 1) / 2 + 0.5
        imgs = np.concatenate(
            [np.concatenate([fake_x[i * 7 + j] for j in range(7)], axis=1)
             for i in range(len(fake_x)//7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.asnumpy())
        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Egy korszak tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Generált példák megjelenítése
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # A szintetikus adatok normalizálása N(0, 1)-re
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))

    optimizer_hp = {"lr": lr, "beta_1": 0.5, "beta_2": 0.999}
    optimizer_D = tf.keras.optimizers.Adam(**optimizer_hp)
    optimizer_G = tf.keras.optimizers.Adam(**optimizer_hp)

    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)

    for epoch in range(1, num_epochs + 1):
        # Egy korszak tanítása
        timer = d2l.Timer()
        metric = d2l.Accumulator(3) # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(mean=0, stddev=1,
                                 shape=(batch_size, 1, 1, latent_dim))
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       d2l.update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)

        # Generált példák megjelenítése
        Z = tf.random.normal(mean=0, stddev=1, shape=(21, 1, 1, latent_dim))
        # A szintetikus adatok normalizálása N(0, 1)-re
        fake_x = net_G(Z) / 2 + 0.5
        imgs = tf.concat([tf.concat([fake_x[i * 7 + j] for j in range(7)],
                                    axis=1)
                          for i in range(len(fake_x) // 7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # A veszteségek megjelenítése
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device._device_name)}')
```

A modellt csupán néhány korszakon tanítjuk demonstrációs célból.
Jobb teljesítményért
a `num_epochs` változó nagyobb számra állítható.

```{.python .input}
#@tab mxnet, pytorch
latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

```{.python .input}
#@tab tensorflow
latent_dim, lr, num_epochs = 100, 0.0005, 40
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

## Összefoglalás

* A DCGAN architektúrája négy konvolúciós réteget tartalmaz a diszkriminátorhoz és négy „törtlépéses" konvolúciós réteget a generátorhoz.
* A diszkriminátor 4 rétegű léptetett konvolúciókból áll batchnormalizációval (kivéve a bemeneti réteget) és szivárgó ReLU aktivációkkal.
* A szivárgó ReLU egy nemlineáris függvény, amely negatív bemenet esetén nem nulla kimenetet ad. Célja a „haldokló ReLU” probléma kijavítása, és segít a gradienseknek könnyebben átfolyni az architektúrán.


## Feladatok

1. Mi történik, ha standard ReLU aktivációt alkalmazunk szivárgó ReLU helyett?
1. Alkalmazd a DCGAN-t a Fashion-MNIST adathalmazra, és nézd meg, melyik kategória működik jól és melyik nem.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/409)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1083)
:end_tab:
