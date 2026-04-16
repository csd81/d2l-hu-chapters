```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Párnázás és lépésköz
:label:`sec_padding`

Idézzük fel a konvolúció példáját a :numref:`fig_correlation`-ban. A bemenet magassága és szélessége egyaránt 3 volt, a konvolúciós kernel magassága és szélessége egyaránt 2, ami $2\times2$ dimenziójú kimeneti reprezentációt eredményezett. Feltételezve, hogy a bemeneti alak $n_\textrm{h}\times n_\textrm{w}$ és a konvolúciós kernel alakja $k_\textrm{h}\times k_\textrm{w}$, a kimeneti alak $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$ lesz: csak addig tudjuk eltolni a konvolúciós kernelt, amíg kifogy a pixelekből, amelyekre a konvolúciót alkalmazni tudjuk.

A következőkben számos technikát fogunk megvizsgálni, beleértve a párnázást és a lépéses konvolúciókat, amelyek nagyobb kontrollt kínálnak a kimenet mérete felett. Motivációképpen megjegyezzük, hogy mivel a kernelek általában 1-nél nagyobb szélességgel és magassággal rendelkeznek, sok egymást követő konvolúció alkalmazása után hajlamosak vagyunk olyan kimeneteket kapni, amelyek jóval kisebbek a bemeneténél. Ha egy $240 \times 240$ pixeles képpel kezdünk, tíz réteg $5 \times 5$-ös konvolúció $200 \times 200$ pixelre csökkenti a képet, levágva az eredeti kép $30\%$-át, és ezzel elpusztítva az eredeti kép határain lévő érdekes információkat. A *párnázás* a legnépszerűbb eszköz ennek a kérdésnek a kezelésére. Más esetekben drasztikusan csökkenteni szeretnénk a dimenzionalitást, például ha az eredeti bemeneti felbontást kezelhetetlennek találjuk. A *lépéses konvolúciók* egy népszerű technika, amely ezekben az esetekben segíthet.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Párnázás

Amint fentebb leírtuk, konvolúciós rétegek alkalmazásakor az egyik trükkös probléma az, hogy hajlamosak vagyunk pixeleket elveszíteni a kép kerületén. Tekintsük a :numref:`img_conv_reuse`-t, amely a pixelhasználatot ábrázolja a konvolúciós kernel méretének és a képen belüli pozíciónak a függvényeként. A sarokban lévő pixeleket alig használják.

![Pixelhasználat $1 \times 1$, $2 \times 2$ és $3 \times 3$ méretű konvolúciókhoz.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Mivel általában kis kerneleket használunk, minden adott konvolúciónál csak néhány pixelt veszíthetünk, de ez összeadódhat, ha sok egymást követő konvolúciós réteget alkalmazunk. Ennek a problémának egy egyszerű megoldása az, hogy extra kitöltő pixeleket adunk a bemeneti kép határai köré, növelve ezzel a kép tényleges méretét. Általában az extra pixelek értékét nullára állítjuk. A :numref:`img_conv_pad`-ban egy $3 \times 3$-as bemenetet párnázunk, növelve méretét $5 \times 5$-re. A megfelelő kimenet ezután $4 \times 4$-es mátrixra növekszik. Az árnyékolt részek az első kimeneti elem, valamint a kimeneti számításhoz használt bemeneti és kernel tenzorelemek: $0\times0+0\times1+0\times2+0\times3=0$.

![Kétdimenziós keresztkorreláció párnázással.](../img/conv-pad.svg)
:label:`img_conv_pad`

Általánosságban, ha összesen $p_\textrm{h}$ sor párnázást adunk hozzá (nagyjából felét felül és felét alul), és összesen $p_\textrm{w}$ oszlop párnázást (nagyjából felét bal oldalon és felét jobb oldalon), a kimeneti alak lesz:

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

Ez azt jelenti, hogy a kimenet magassága és szélessége rendre $p_\textrm{h}$-val és $p_\textrm{w}$-vel növekszik.

Sok esetben $p_\textrm{h}=k_\textrm{h}-1$ és $p_\textrm{w}=k_\textrm{w}-1$ értékeket szeretnénk beállítani, hogy a bemenetnek és a kimenetnek azonos magassága és szélessége legyen. Ez megkönnyíti az egyes rétegek kimeneti alakjának előrejelzését a hálózat megalkotásakor. Feltételezve, hogy $k_\textrm{h}$ itt páratlan, $p_\textrm{h}/2$ sort párnázunk a magasság mindkét oldalán. Ha $k_\textrm{h}$ páros, az egyik lehetőség $\lceil p_\textrm{h}/2\rceil$ sort párnázni a bemenet tetejére és $\lfloor p_\textrm{h}/2\rfloor$ sort az aljára. A szélességet ugyanígy fogjuk párnázni mindkét oldalon.

A CNN-ek általában páratlan magasságú és szélességű konvolúciós kerneleket használnak, mint például 1, 3, 5 vagy 7. A páratlan kernelméretek megválasztásának az az előnye, hogy megőrizhetjük a dimenzionalitást, miközben azonos számú sorral párnázunk felül és alul, és azonos számú oszloppal bal és jobb oldalon.

Ráadásul a páratlan kernelek és a párnázás ilyen használata a dimenzionalitás pontos megőrzéséhez adminisztrációs előnnyel jár. Bármely kétdimenziós `X` tenzor esetén, ha a kernel mérete páratlan és a párnázási sorok és oszlopok száma minden oldalon azonos, ezzel azonos magasságú és szélességű kimenetet produkálva, mint a bemenet, tudjuk, hogy az `Y[i, j]` kimenet az `X[i, j]`-re középpontosított ablakkal számított bemeneti és konvolúciós kernel keresztkorrelációjaként kerül kiszámításra.

A következő példában egy 3-as magasságú és szélességű kétdimenziós konvolúciós réteget hozunk létre, és (**1 pixel párnázást alkalmazunk minden oldalon.**) Egy 8-as magasságú és szélességű bemenet esetén azt találjuk, hogy a kimenet magassága és szélessége szintén 8.

```{.python .input}
%%tab mxnet
# Segédfüggvényt definiálunk a konvolúciók kiszámításához. Inicializálja
# a konvolúciós réteg súlyait, és elvégzi a megfelelő dimenzióbővítéseket
# és -csökkentéseket a bemeneten és a kimeneten
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) jelzi, hogy a kötegméret és a csatornák száma egyaránt 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Az első két dimenzió eltávolítása: minták és csatornák
    return Y.reshape(Y.shape[2:])

# Mindkét oldalon 1 sor és oszlop párnázást adunk hozzá, összesen 2 sort vagy oszlopot
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# Segédfüggvényt definiálunk a konvolúciók kiszámításához. Inicializálja
# a konvolúciós réteg súlyait, és elvégzi a megfelelő dimenzióbővítéseket
# és -csökkentéseket a bemeneten és a kimeneten
def comp_conv2d(conv2d, X):
    # (1, 1) jelzi, hogy a kötegméret és a csatornák száma egyaránt 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Az első két dimenzió eltávolítása: minták és csatornák
    return Y.reshape(Y.shape[2:])

# Mindkét oldalon 1 sor és oszlop párnázást adunk hozzá, összesen 2 sort
# vagy oszlopot
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# Segédfüggvényt definiálunk a konvolúciók kiszámításához. Inicializálja
# a konvolúciós réteg súlyait, és elvégzi a megfelelő dimenzióbővítéseket
# és -csökkentéseket a bemeneten és a kimeneten
def comp_conv2d(conv2d, X):
    # (1, 1) jelzi, hogy a kötegméret és a csatornák száma egyaránt 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Az első két dimenzió eltávolítása: minták és csatornák
    return tf.reshape(Y, Y.shape[1:3])
# Mindkét oldalon 1 sor és oszlop párnázást adunk hozzá, összesen 2 sort
# vagy oszlopot
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# Segédfüggvényt definiálunk a konvolúciók kiszámításához. Inicializálja
# a konvolúciós réteg súlyait, és elvégzi a megfelelő dimenzióbővítéseket
# és -csökkentéseket a bemeneten és a kimeneten
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1) jelzi, hogy a kötegméret és a csatornák száma egyaránt 1
    key = jax.random.PRNGKey(d2l.get_seed())
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # A dimenziók eltávolítása: minták és csatornák
    return Y.reshape(Y.shape[1:3])
# Mindkét oldalon 1 sor és oszlop párnázást adunk hozzá, összesen 2 sort vagy oszlopot
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Amikor a konvolúciós kernel magassága és szélessége különböző, azonos magasságú és szélességű kimenetet és bemenetet kaphatunk [**különböző párnázási számok beállításával a magassághoz és szélességhez.**]

```{.python .input}
%%tab mxnet
# 5 magasságú és 3 szélességű konvolúciós kernelt használunk. A párnázás
# mindkét oldalán a magassághoz 2, a szélességhez 1
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# 5 magasságú és 3 szélességű konvolúciós kernelt használunk. A párnázás
# mindkét oldalán a magassághoz 2, a szélességhez 1
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# 5 magasságú és 3 szélességű konvolúciós kernelt használunk. A párnázás
# mindkét oldalán a magassághoz 2, a szélességhez 1
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# 5 magasságú és 3 szélességű konvolúciós kernelt használunk. A párnázás
# mindkét oldalán a magassághoz 2, a szélességhez 1
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## Lépésköz

A keresztkorreláció számításakor a bemeneti tenzor bal felső sarkától indítjuk a konvolúciós ablakot, majd végigcsúsztatjuk lefelé és jobbra is az összes helyen. Az előző példákban alapértelmezés szerint egyszerre egy elemet csúsztattunk. Azonban néha, akár számítási hatékonyság, akár a mintavételezés csökkentése érdekében, egyszerre egynél több elemet mozgatjuk az ablakot, kihagyva a közbenső helyeket. Ez különösen hasznos, ha a konvolúciós kernel nagy, mivel a mögöttes kép nagy területét fedi le.

Az egy csúsztatásonként bejárt sorok és oszlopok számát *lépésköznek* nevezzük. Eddig 1-es lépésközt használtunk mind a magassághoz, mind a szélességhez. Néha nagyobb lépésközt szeretnénk használni. A :numref:`img_conv_stride` egy kétdimenziós keresztkorreláció műveletet mutat 3-as függőleges és 2-es vízszintes lépésközzel. Az árnyékolt részek a kimeneti elemek, valamint a kimeneti számításhoz használt bemeneti és kernel tenzorelemek: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$. Láthatjuk, hogy amikor az első oszlop második eleme kerül kiszámításra, a konvolúciós ablak három sorral csúszik le. A konvolúciós ablak két oszloppal csúszik jobbra, amikor az első sor második eleme kerül kiszámításra. Amikor a konvolúciós ablak tovább csúszik két oszloppal jobbra a bemeneten, nincs kimenet, mert a bemeneti elem nem tölti ki az ablakot (hacsak nem adunk hozzá egy másik oszlop párnázást).

![Keresztkorreláció 3-as és 2-es lépésközzel a magassághoz és szélességhez.](../img/conv-stride.svg)
:label:`img_conv_stride`

Általánosságban, ha a magasság lépésköze $s_\textrm{h}$ és a szélesség lépésköze $s_\textrm{w}$, a kimeneti alak:

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

Ha $p_\textrm{h}=k_\textrm{h}-1$ és $p_\textrm{w}=k_\textrm{w}-1$ értékeket állítunk be, akkor a kimeneti alak egyszerűsíthető $\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$-ra. Egy lépéssel tovább haladva, ha a bemeneti magasság és szélesség osztható a magassági és szélességi lépésközökkel, akkor a kimeneti alak $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$ lesz.

Az alábbiakban [**a magassághoz és szélességhez egyaránt 2-es lépésközt állítunk be**], ezzel felezve a bemeneti magasságot és szélességet.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

Nézzünk meg (**egy kissé bonyolultabb példát**).

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## Összefoglalás és vita

A párnázás növelheti a kimenet magasságát és szélességét. Ez gyakran arra használatos, hogy a kimenetnek azonos magassága és szélessége legyen, mint a bemenetnek, elkerülve a kimenet nemkívánatos zsugorodását. Ráadásul biztosítja, hogy az összes pixel egyforma gyakorisággal kerüljön felhasználásra. Általában szimmetrikus párnázást választunk a bemeneti magasság és szélesség mindkét oldalán. Ebben az esetben $(p_\textrm{h}, p_\textrm{w})$ párnázásra hivatkozunk. Leggyakrabban $p_\textrm{h} = p_\textrm{w}$-t állítunk be, ekkor egyszerűen azt mondjuk, hogy $p$ párnázást választunk.

Hasonló konvenció érvényes a lépésközre is. Amikor a vízszintes $s_\textrm{h}$ és függőleges $s_\textrm{w}$ lépésköz megegyezik, egyszerűen $s$ lépésközről beszélünk. A lépésköz csökkentheti a kimenet felbontását, például $n > 1$ esetén a kimenet magasságát és szélességét a bemenet magasságának és szélességének csak $1/n$-ére csökkentve. Alapértelmezés szerint a párnázás 0 és a lépésköz 1.

Eddig tárgyalt összes párnázás egyszerűen nullákkal terjesztette ki a képeket. Ennek jelentős számítási előnye van, mivel triviálisan elvégezhető. Ráadásul az operátorok úgy tervezhetők, hogy implicit módon kihasználják ezt a párnázást anélkül, hogy további memóriát kellene lefoglalni. Ugyanakkor lehetővé teszi a CNN-ek számára, hogy implicit pozícióinformációkat kódoljanak a képen belül, egyszerűen azzal, hogy megtanulják, hol van az "üres terület". Számos alternatíva létezik a nullás párnázással szemben. :citet:`Alsallakh.Kokhlikyan.Miglani.ea.2020` átfogó áttekintést nyújtott ezekről (bár anélkül, hogy egyértelmű útmutatást adna arra vonatkozóan, mikor kell nem nulla párnázásokat használni, hacsak nem lépnek fel műtermékek).


## Feladatok

1. Adott ennek a résznek az utolsó kódpéldája $(3, 5)$-ös kernelmérettel, $(0, 1)$ párnázással és $(3, 4)$-es lépésközzel, számítsuk ki a kimeneti alakot, hogy ellenőrizzük, megegyezik-e a kísérleti eredménnyel.
1. Audiójelek esetén minek felel meg a 2-es lépésköz?
1. Implementáljunk tükrözéses párnázást, azaz olyan párnázást, ahol a szélső értékeket egyszerűen tükrözik a tenzorok kiterjesztéséhez.
1. Milyen számítási előnyei vannak az 1-nél nagyobb lépésköznek?
1. Milyen statisztikai előnyei lehetnek az 1-nél nagyobb lépésköznek?
1. Hogyan implementálnánk $\frac{1}{2}$-es lépésközt? Minek felel ez meg? Mikor lenne hasznos?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17997)
:end_tab:
