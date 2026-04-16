```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Konvolúciós Hálózati Architektúrák Tervezése
:label:`sec_cnn-design`

Az előző szakaszok áttekintést nyújtottak a modern hálózattervezésről a számítógépes látás területén. Minden általunk tárgyalt munka közös jellemzője, hogy nagyban támaszkodott a kutatók intuíciójára. Sok architektúrát erősen az emberi kreativitás ihletett, és jóval kevésbé a mélyhálózatok által kínált tervezési tér szisztematikus feltárása. Ennek ellenére ez a *hálózatmérnöki* megközelítés rendkívül sikeres volt. 

Azóta, hogy az AlexNet (:numref:`sec_alexnet`)
legyőzte a hagyományos számítógépes látási modelleket az ImageNet-en,
népszerűvé vált nagyon mély hálózatok építése
konvolúciós blokkok egymásra halmozásával, mindegyik ugyanazon minta szerint tervezve.
Különösen a $3 \times 3$ konvolúciókat
a VGG hálózatok népszerűsítették (:numref:`sec_vgg`).
A NiN (:numref:`sec_nin`) megmutatta, hogy még a $1 \times 1$ konvolúciók is
hasznosak lehetnek lokális nemlinearitások hozzáadásával.
Ráadásul a NiN megoldotta az információ aggregálásának problémáját a hálózat fejénél
az összes helyszínen történő aggregálással.
A GoogLeNet (:numref:`sec_googlenet`) különböző konvolúciós szélességű több ágat adott hozzá,
összekombinálva a VGG és NiN előnyeit az Inception blokkjában.
A ResNet-ek (:numref:`sec_resnet`)
megváltoztatták az induktív torzítást az identitás leképezés felé (a $f(x) = 0$-tól). Ez lehetővé tette nagyon mély hálózatok létrehozását. Majdnem egy évtizeddel később a ResNet tervezés még mindig népszerű, ami a tervezés időtállóságának bizonyítéka. Végül a ResNeXt (:numref:`subsec_resnext`) csoportosított konvolúciókat adott hozzá, jobb kompromisszumot kínálva a paraméterek és számítás között. A látási Transformerek előfutáraként, a Squeeze-and-Excitation Networks (SENets) lehetővé teszik hatékony információátvitelt helyszínek között
:cite:`Hu.Shen.Sun.2018`. Ezt csatornánkénti globális figyelmi függvény számításával érték el. 

Eddig kihagytuk a *neurális architektúra keresés* (NAS) által nyert hálózatokat :cite:`zoph2016neural,liu2018darts`. Ezt azért tettük, mert költségük általában óriási, brutális erővel történő keresésre, genetikus algoritmusokra, megerősítéses tanulásra vagy valamilyen más hiperparaméter-optimalizálási formára támaszkodnak. Adott keresési tér esetén
a NAS keresési stratégiát használ az architektúra automatikus kiválasztására
a visszaadott teljesítménybecslés alapján.
A NAS eredménye
egyszeri hálózatpéldány. Az EfficientNet-ek ennek a keresésnek a figyelemre méltó eredményei :cite:`tan2019efficientnet`.

A következőkben egy olyan ötletet tárgyalunk, amely teljesen eltér a *legjobb egyetlen hálózat* keresésétől. Viszonylag olcsó számítási szempontból, tudományos betekintéseket nyújt útközben, és meglehetősen hatékony az eredmények minősége szempontjából. Tekintsük át a :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` által javasolt stratégiát a *hálózattervezési terek tervezésére*. Ez a stratégia kombinálja a kézi tervezés és a NAS erősségeit. Ezt úgy éri el, hogy *hálózatok eloszlásain* működik és optimalizálja az eloszlásokat úgy, hogy jó teljesítményt érjen el teljes hálózatcsaládokra. Ennek eredménye a *RegNet-ek*, konkrétan a RegNetX és RegNetY, plusz egy sor irányelv a teljesítményes CNN-ek tervezéséhez.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## Az AnyNet Tervezési Tér
:label:`subsec_the-anynet-design-space`

Az alábbi leírás szorosan követi a :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` érvelését, néhány rövidítéssel, hogy illeszkedjen a könyv terjedelméhez.
Először is szükségünk van egy sablonra a vizsgálandó hálózatcsaládhoz. A fejezetben tárgyalt tervezések egyik közös vonása, hogy a hálózatok egy *törzsből* (*stem*), egy *testből* (*body*) és egy *fejből* (*head*) állnak. A törzs elvégzi a kezdeti képfeldolgozást, általában nagyobb ablakméretű konvolúciókkal. A test több blokkból áll, amelyek elvégzik az összes szükséges transzformáció nagy részét a nyers képektől az objektumreprezentációkig. Végül a fej a kívánt kimenetté alakítja ezt, például többosztályos osztályozás esetén softmax regresszorral.
A test viszont több szakaszból áll, amelyek egyre csökkenő felbontáson dolgozzák fel a képet. Valójában mind a törzs, mind az egymást követő szakaszok negyedére csökkentik a térbeli felbontást. Végül minden szakasz egy vagy több blokkból áll. Ez a minta közös az összes hálózatban, a VGG-től a ResNeXt-ig. A generikus AnyNet hálózatok tervezéséhez :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` a :numref:`fig_resnext_block` ResNeXt blokkját használta.


![Az AnyNet tervezési tér. Az egyes nyilak mentén szereplő $(\mathit{c}, \mathit{r})$ számok a csatornák $c$ számát és a képek $\mathit{r} \times \mathit{r}$ felbontását jelzik az adott ponton. Balról jobbra: törzs, test és fej által alkotott általános hálózatszerkezet; négy szakaszból álló test; egy szakasz részletes szerkezete; két alternatív blokkszerkezet, egyik almintavételezés nélkül, a másik minden dimenzióban felezi a felbontást. A tervezési döntések magukban foglalják a mélységet $\mathit{d_i}$, a kimeneti csatornák számát $\mathit{c_i}$, a csoportok számát $\mathit{g_i}$ és a szűk keresztmetszet arányát $\mathit{k_i}$ bármely $\mathit{i}$ szakasz esetén.](../img/anynet.svg)
:label:`fig_anynet_full`

Tekintsük át részletesen a :numref:`fig_anynet_full` által felvázolt szerkezetet. Mint említettük, egy AnyNet egy törzsből, testből és fejből áll. A törzs bemeneteként RGB képeket fogad (3 csatorna), $2$-es lépésközű $3 \times 3$-as konvolúciót alkalmaz, amelyet batch normalizáció követ, hogy a felbontást $r \times r$-ről $r/2 \times r/2$-re csökkentse. Emellett $c_0$ csatornát állít elő, amelyek a test bemenetéül szolgálnak.

Mivel a hálózatot $224 \times 224 \times 3$ alakú ImageNet képekkel való hatékony működésre tervezték, a test feladata ennek $7 \times 7 \times c_4$-re való csökkentése 4 szakaszon keresztül (felidézve, hogy $224 / 2^{1+4} = 7$), mindegyik szakasznál végső lépésközzel $2$. Végül a fej teljesen standard tervezést alkalmaz globális átlagos összesítésen keresztül, hasonlóan a NiN-hez (:numref:`sec_nin`), amelyet egy teljesen összekötött réteg követ, amely $n$-dimenziós vektort bocsát ki $n$-osztályos osztályozáshoz.

A legtöbb releváns tervezési döntés a hálózat testéhez kötődik. Ez szakaszonként halad, ahol minden szakasz azonos típusú ResNeXt blokkokból áll, ahogyan azt a :numref:`subsec_resnext` részben tárgyaltuk. A tervezés ott is teljesen általános: egy olyan blokkal kezdünk, amely $2$-es lépésközzel felezi a felbontást (a :numref:`fig_anynet_full` jobb szélső blokkja). Ennek megfeleltetéséhez a ResNeXt blokk reziduális ágának át kell haladnia egy $1 \times 1$-es konvolúción. Ezt a blokkot változó számú további ResNeXt blokk követi, amelyek mind a felbontást, mind a csatornák számát változatlanul hagyják. Megjegyzendő, hogy bevett tervezési gyakorlat egy enyhe szűk keresztmetszet hozzáadása a konvolúciós blokkok tervezésekor.
Ennélfogva a $k_i \geq 1$ szűk keresztmetszet arányával bizonyos számú csatornát, $c_i/k_i$-t biztosítunk az $i$ szakasz minden blokkján belül (ahogyan a kísérletek mutatják, ez nem igazán hatékony, ezért kihagyható). Végül, mivel ResNeXt blokkokkal dolgozunk, ki kell választanunk a $g_i$ csoportok számát is a csoportosított konvolúciókhoz az $i$ szakaszban.

Ez a látszólag általános tervezési tér mégis sok paramétert biztosít számunkra: beállíthatjuk a blokk szélességet (csatornák száma) $c_0, \ldots c_4$, a mélységet (blokkok száma) szakaszonként $d_1, \ldots d_4$, a szűk keresztmetszet arányokat $k_1, \ldots k_4$, és a csoport szélességeket (csoportok száma) $g_1, \ldots g_4$.
Mindez összesen 17 paramétert jelent, ami indokolatlanul nagy számú konfigurációt eredményez, amelyeket érdemes lenne megvizsgálni. Szükségünk van néhány eszközre, hogy ezt a hatalmas tervezési teret hatékonyan csökkentsük. Ez az, ahol a tervezési terek fogalmi szépsége megmutatkozik. Mielőtt ezt megtennénk, először implementáljuk az általános tervezést.

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

Minden szakasz `depth` darab ResNeXt blokkból áll,
ahol a `num_channels` adja meg a blokk szélességét.
Megjegyzendő, hogy az első blokk felezi a bemeneti képek magasságát és szélességét.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

A hálózat törzsének, testének és fejének összeillesztésével
befejezzük az AnyNet implementációját.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## Tervezési Terek Eloszlásai és Paraméterei

Ahogyan a :numref:`subsec_the-anynet-design-space` részben tárgyaltuk, egy tervezési tér paraméterei az abban a tervezési térben lévő hálózatok hiperparaméterei.
Vizsgáljuk meg a jó paraméterek azonosításának problémáját az AnyNet tervezési térben. Megpróbálhatnánk megtalálni az *egyetlen legjobb* paraméterválasztást egy adott számítási mennyiséghez (pl. FLOPs és számítási idő). Ha csak *két* lehetséges választást engednénk meg minden paraméterre, $2^{17} = 131072$ kombinációt kellene megvizsgálnunk a legjobb megoldás megtalálásához. Ez egyértelműen kivitelezhetetlen a rendkívüli költsége miatt. Ráadásul semmit sem tanulunk ebből a gyakorlatból abban a tekintetben, hogyan kellene hálózatot tervezni. Következő alkalommal, amikor hozzáadunk mondjuk egy X-szakaszt, egy eltolási műveletet vagy valami hasonlót, elölről kellene kezdenünk. Sőt, a tanítás véletlenszerűsége miatt (kerekítés, keverés, bithiba) valószínűleg két futtatás sem fog pontosan ugyanolyan eredményt produkálni. Jobb stratégia lenne általános irányelveket meghatározni arra vonatkozóan, hogyan kellene összefüggeniük a paraméterválasztásoknak. Például a szűk keresztmetszet aránya, a csatornák, blokkok, csoportok száma, vagy ezek rétegek közötti változása ideálisan egyszerű szabályok összességével lenne meghatározott. A :citet:`radosavovic2019network` megközelítése a következő négy feltevésre támaszkodik:

1. Feltételezzük, hogy általános tervezési elvek valóban léteznek, így sok, ezeket a követelményeket kielégítő hálózatnak jó teljesítményt kellene nyújtania. Következésképpen egy hálózatokon értelmezett *eloszlás* azonosítása ésszerű stratégia lehet. Más szóval, feltételezzük, hogy sok jó tű van a szénakazalban.
1. Nem szükséges konvergenciáig tanítani a hálózatokat, mielőtt megítélhetnénk, hogy egy hálózat jó-e. Ehelyett elegendő a közbenső eredményeket megbízható útmutatóként használni a végső pontossághoz. A cél optimalizálásához (közelítő) helyettesítők használatát multi-fidelity optimalizálásnak nevezzük :cite:`forrester2007multi`. Következésképpen a tervezési optimalizálás az adathalmazon való csupán néhány átmenet után elért pontosság alapján hajtható végre, ami jelentősen csökkenti a költséget.
1. Kisebb méretben (kisebb hálózatoknál) kapott eredmények általánosíthatók nagyobbakra. Következésképpen az optimalizálást strukturálisan hasonló, de kevesebb blokkal, kevesebb csatornával stb. rendelkező hálózatokon végzik. Csak a végén kell ellenőrizni, hogy az így talált hálózatok nagy méretben is jó teljesítményt nyújtanak-e.
1. A tervezés szempontjai közelítőleg faktorizálhatók, így lehetséges hatásukat a kimenet minőségére bizonyos mértékig egymástól függetlenül következtetni. Más szóval, az optimalizálási probléma mérsékelten könnyű.

Ezek a feltevések lehetővé teszik számunkra, hogy sok hálózatot olcsón teszteljünk. Különösen *mintát vehetünk* egyenletesen a konfigurációk teréből, és kiértékelhetjük teljesítményüket. Ezt követően a paraméterválasztás minőségét a szóban forgó hálózatokkal elérhető hiba/pontosság *eloszlásának* áttekintésével értékelhetjük. Jelölje $F(e)$ a kumulatív eloszlásfüggvényt (CDF) egy adott tervezési tér hálózatai által elkövetett hibákra, $p$ valószínűségi eloszlással mintázva. Azaz:

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

Célunk most egy $p$ eloszlás megtalálása a *hálózatokon* úgy, hogy a legtöbb hálózatnak nagyon alacsony hibaráta legyen, és ahol $p$ tartóhalmaza tömör. Természetesen ezt pontosan számítani nem lehetséges. Hálózatok $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ mintájához folyamodunk (rendre $e_1, \ldots, e_n$ hibákkal) a $p$-ből, és az empirikus CDF-et $\hat{F}(e, \mathcal{Z})$ használjuk helyette:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

Amikor az egyik választáskészlet CDF-je dominálja (vagy egyenlő) egy másik CDF-et, következik, hogy az adott paraméterválasztás fölényesebb (vagy közömbös). Ennek megfelelően :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` megosztott hálózati szűk keresztmetszet arányokkal $k_i = k$ kísérletezett a hálózat minden $i$ szakaszára. Ez megszünteti a szűk keresztmetszet arányt szabályozó négy paraméter közül hármat. Annak értékeléséhez, hogy ez (negatívan) befolyásolja-e a teljesítményt, hálózatokat lehet mintázni a korlátozott és a korlátlan eloszlásból, majd összehasonlítani a megfelelő CDF-eket. Kiderül, hogy ez a korlát egyáltalán nem befolyásolja a hálózatok eloszlásának pontosságát, ahogyan az a :numref:`fig_regnet-fig` első paneljén látható.
Hasonlóképpen dönthetünk úgy, hogy ugyanazt a csoportszélességet $g_i = g$ választjuk a hálózat különböző szakaszainál. Ez sem befolyásolja a teljesítményt, ahogyan az a :numref:`fig_regnet-fig` második paneljén látható.
Mindkét lépés együttesen hat szabad paraméterrel csökkenti a paraméterek számát.

![Tervezési terek empirikus hibaeloszlás-függvényeinek összehasonlítása. $\textrm{AnyNet}_\mathit{A}$ az eredeti tervezési tér; $\textrm{AnyNet}_\mathit{B}$ összeköti a szűk keresztmetszet arányokat, $\textrm{AnyNet}_\mathit{C}$ a csoportszélességeket is összeköti, $\textrm{AnyNet}_\mathit{D}$ növeli a hálózat mélységét a szakaszokon át. Balról jobbra: (i) a szűk keresztmetszet arányok összekapcsolása nem befolyásolja a teljesítményt; (ii) a csoportszélességek összekapcsolása nem befolyásolja a teljesítményt; (iii) a hálózati szélességek (csatornák) növelése a szakaszokon át javítja a teljesítményt; (iv) a hálózati mélységek növelése a szakaszokon át javítja a teljesítményt. Az ábra forrása: :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

Ezután módszereket keresünk a szakaszok szélességére és mélységére vonatkozó számos lehetséges választás csökkentésére. Ésszerű feltételezés, hogy ahogy mélyebbre megyünk, a csatornák számának növekednie kell, azaz $c_i \geq c_{i-1}$ ($w_{i+1} \geq w_i$ a :numref:`fig_regnet-fig` jelölésük szerint), ami $\textrm{AnyNetX}_D$-t eredményez. Hasonlóképpen ugyanolyan ésszerű feltételezni, hogy ahogy a szakaszok előrehaladnak, mélyebbé kell válniuk, azaz $d_i \geq d_{i-1}$, ami $\textrm{AnyNetX}_E$-t eredményez. Ez kísérletileg ellenőrizhető a :numref:`fig_regnet-fig` harmadik és negyedik paneljén.

## RegNet

Az eredményül kapott $\textrm{AnyNetX}_E$ tervezési tér egyszerű hálózatokból áll,
amelyek könnyen értelmezhető tervezési elveket követnek:

* Megosztott szűk keresztmetszet arány $k_i = k$ minden $i$ szakaszra;
* Megosztott csoportszélesség $g_i = g$ minden $i$ szakaszra;
* A hálózati szélesség növelése a szakaszokon át: $c_{i} \leq c_{i+1}$;
* A hálózati mélység növelése a szakaszokon át: $d_{i} \leq d_{i+1}$.

Ez bennünket egy végső döntéskészlettel hagy: hogyan válasszuk meg a fenti paraméterek konkrét értékeit a végső $\textrm{AnyNetX}_E$ tervezési térhez. A $\textrm{AnyNetX}_E$ eloszlásból legjobban teljesítő hálózatok tanulmányozásával a következő figyelhető meg: a hálózat szélessége ideálisan lineárisan növekszik a blokkindexszel a hálózaton át, azaz $c_j \approx c_0 + c_a j$, ahol $j$ a blokkindex és a meredekség $c_a > 0$. Mivel csak szakaszonként választhatunk különböző blokkszélességet, darabonként konstans függvényhez jutunk, amelyet ennek a függőségnek a megfeleltetésére terveztek. Sőt, a kísérletek azt is mutatják, hogy $k = 1$ szűk keresztmetszet arány teljesít a legjobban, azaz tanácsos egyáltalán nem használni szűk keresztmetszeteket.

Az érdeklődő olvasónak ajánljuk, hogy tekintse át a különböző számítási mennyiségekhez tervezett konkrét hálózatok részleteit a :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` alapos olvasásával. Például egy hatékony 32 rétegű RegNetX változatot $k = 1$ (szűk keresztmetszet nélkül), $g = 16$ (csoportszélesség 16), az első és második szakasz esetén rendre $c_1 = 32$ és $c_2 = 80$ csatornával, $d_1=4$ és $d_2=6$ blokk mélységgel határoznak meg. A tervezés megdöbbentő tanulsága, hogy még nagyobb méretű hálózatok vizsgálatakor is érvényes marad. Sőt, még a globális csatornaaktiválással rendelkező Squeeze-and-Excitation (SE) hálózati tervezésekre (RegNetY) is igaz :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

Láthatjuk, hogy minden RegNetX szakasz fokozatosan csökkenti a felbontást és növeli a kimeneti csatornák számát.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## Tanítás

A 32 rétegű RegNetX tanítása a Fashion-MNIST adathalmazon ugyanolyan, mint korábban.

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## Megvitatás

A látáshoz kívánatos induktív torzításoknak (feltételezések vagy preferenciák), mint a lokalitás és a transzlációs invariancia (:numref:`sec_why-conv`), köszönhetően a CNN-ek uralták ezt a területet. Ez így maradt a LeNet-től egészen addig, amíg a Transformerek (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` el nem kezdték felülmúlni a CNN-eket pontosság tekintetében. Bár a látási Transformereket érintő legújabb fejlesztések nagy része *visszaportolható* a CNN-ekbe :cite:`liu2022convnet`, ez csak magasabb számítási költséggel lehetséges. Ugyanilyen fontos, hogy a legújabb hardveres optimalizációk (NVIDIA Ampere és Hopper) csak szélesítették a Transformerek javára mutatkozó szakadékot.

Megjegyzendő, hogy a Transformereknek lényegesen alacsonyabb fokú induktív torzításuk van a lokalitás és a transzlációs invariancia felé, mint a CNN-eknek. A tanult struktúrák győzelme nem utolsósorban a nagy képgyűjtemények elérhetőségének köszönhető, mint a LAION-400m és a LAION-5B :cite:`schuhmann2022laion`, amely akár 5 milliárd képet is tartalmaz. Meglepő módon, az e kontextusban legjelentősebb munkák egy része MLP-ket is magában foglal :cite:`tolstikhin2021mlp`.

Összefoglalva, a látási Transformerek (:numref:`sec_vision-transformer`) mára vezető pozíciót foglalnak el
a nagy léptékű képosztályozás legkorszerűbb teljesítménye tekintetében,
megmutatva, hogy *a skálázhatóság felülírja az induktív torzításokat* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Ez magában foglalja a nagy léptékű Transformerek előtanítását (:numref:`sec_large-pretraining-transformers`) többfejű önfigyelemmel (:numref:`sec_multihead-attention`). Javasoljuk, hogy merülj bele ezekbe a fejezetekbe a részletesebb tárgyalásért.

## Feladatok

1. Növeld a szakaszok számát négyre. Tudsz-e mélyebb RegNetX-et tervezni, amely jobban teljesít?
1. "De-ResNeXt-esítsd" a RegNet-eket úgy, hogy a ResNeXt blokkot ResNet blokkal cseréled le. Hogyan teljesít az új modelled?
1. Implementáld a "VioNet" család több példányát a RegNetX tervezési elveit *megsértve*. Hogyan teljesítenek? Melyik tényező ($d_i$, $c_i$, $g_i$, $b_i$) a legfontosabb?
1. Célod a "tökéletes" MLP megtervezése. Tudod-e a fentebb bemutatott tervezési elveket jó architektúrák megtalálásához használni? Lehetséges-e kis hálózatokról nagy hálózatokra extrapolálni?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18009)
:end_tab:

