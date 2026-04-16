```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Több bemeneti és több kimeneti csatorna
:label:`sec_channels`

Bár leírtuk az egyes képeket alkotó több csatornát (például a színes képeknek vannak standard RGB csatornáik a piros, zöld és kék mennyiségének jelzésére) és a :numref:`subsec_why-conv-channels`-ban a több csatornához tartozó konvolúciós rétegeket, eddig numerikus példáinkat egyszerűsítettük azzal, hogy egyetlen bemeneti és egyetlen kimeneti csatornával dolgoztunk. Ez lehetővé tette számunkra, hogy bemeneteinket, konvolúciós kerneleinket és kimeneteinket kétdimenziós tenzorokként képzeljük el.

Amikor csatornákat adunk a képlethez, mind a bemeneteink, mind a rejtett reprezentációink háromdimenziós tenzorokká válnak. Például minden RGB bemeneti kép alakja $3\times h\times w$. Erre a 3-as méretű tengelyre *csatorna* dimenzióként hivatkozunk. A csatornák fogalma olyan régi, mint maguk a CNN-ek: például a LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` is alkalmazza őket. Ebben a részben mélyebben megvizsgáljuk a több bemeneti és több kimeneti csatornát tartalmazó konvolúciós kerneleket.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Több bemeneti csatorna

Amikor a bemeneti adatok több csatornát tartalmaznak, olyan konvolúciós kernelt kell felépítenünk, amelynek ugyanannyi bemeneti csatornája van, mint a bemeneti adatoknak, hogy keresztkorrelációt végezhessen a bemeneti adatokkal. Feltételezve, hogy a bemeneti adatok csatornáinak száma $c_\textrm{i}$, a konvolúciós kernel bemeneti csatornáinak számának is $c_\textrm{i}$-nek kell lennie. Ha a konvolúciós kernel ablakmérete $k_\textrm{h}\times k_\textrm{w}$, akkor $c_\textrm{i}=1$ esetén a konvolúciós kernelünket egyszerűen $k_\textrm{h}\times k_\textrm{w}$ alakú kétdimenziós tenzorként képzelhetjük el.

Azonban, ha $c_\textrm{i}>1$, olyan kernelre van szükségünk, amely *minden* bemeneti csatornához tartalmaz egy $k_\textrm{h}\times k_\textrm{w}$ alakú tenzort. Ezeknek a $c_\textrm{i}$ tenzornak az összefűzése egy $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ alakú konvolúciós kernelt eredményez. Mivel a bemeneti és a konvolúciós kernel egyaránt $c_\textrm{i}$ csatornával rendelkezik, minden csatornánál elvégezhetjük a keresztkorreláció műveletet a bemenet kétdimenziós tenzorján és a konvolúciós kernel kétdimenziós tenzorján, majd összeadhatjuk a $c_\textrm{i}$ eredményt (összegezve a csatornák felett), hogy kétdimenziós tenzort kapjunk. Ez egy többcsatornás bemenet és egy több bemeneti csatornájú konvolúciós kernel közötti kétdimenziós keresztkorreláció eredménye.

A :numref:`fig_conv_multi_in` két bemeneti csatornával végzett kétdimenziós keresztkorreláció példáját mutatja. Az árnyékolt részek az első kimeneti elem, valamint a kimeneti számításhoz használt bemeneti és kernel tenzorelemek: $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Keresztkorreláció számítás két bemeneti csatornával.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


Annak érdekében, hogy valóban megértsük, mi történik itt, (**mi magunk is implementálhatjuk a keresztkorreláció műveleteket több bemeneti csatornával**). Vegyük figyelembe, hogy mindössze annyit csinálunk, hogy csatornánként végzünk keresztkorreláció műveletet, majd összeadjuk az eredményeket.

```{.python .input}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # Először végigiterálunk K 0. dimenzióján (csatorna), majd összeadjuk
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # Először végigiterálunk K 0. dimenzióján (csatorna), majd összeadjuk
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

A :numref:`fig_conv_multi_in`-beli értékeknek megfelelő `X` bemeneti tenzort és `K` kernel tenzort felépíthetjük, hogy (**érvényesítsük a kimenetét**) a keresztkorreláció műveletnek.

```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Több kimeneti csatorna
:label:`subsec_multi-output-channels`

Függetlenül a bemeneti csatornák számától, eddig mindig egy kimeneti csatornát kaptunk. Azonban, ahogyan a :numref:`subsec_why-conv-channels`-ban tárgyaltuk, kiderül, hogy alapvető fontosságú, hogy minden rétegben több csatorna legyen. A legnépszerűbb neurális hálózati architektúrákban valójában növeljük a csatorna dimenziót, ahogy mélyebbre haladunk a neurális hálózatban, általában lecsökkentve a térbeli felbontást a nagyobb *csatorna mélység* érdekében. Intuitívan úgy gondolhatunk minden csatornára, mint amely különböző jellemzőkészletekre reagál. A valóság azonban ennél kissé bonyolultabb. Naiv értelmezés azt sugallná, hogy a reprezentációk pixelenként vagy csatornánként, egymástól függetlenül tanulódnak meg. Ehelyett a csatornák úgy kerülnek optimalizálásra, hogy együttesen hasznosak legyenek. Ez azt jelenti, hogy ahelyett, hogy egyetlen csatornát feleltetnénk meg egy éldetektornak, lehet, hogy egyszerűen azt jelenti, hogy a csatornatér egy iránya az élek detektálásának felel meg.

Jelöljük $c_\textrm{i}$-vel és $c_\textrm{o}$-val rendre a bemeneti és kimeneti csatornák számát, és $k_\textrm{h}$-val és $k_\textrm{w}$-vel a kernel magasságát és szélességét. Több csatornát tartalmazó kimenet eléréséhez *minden* kimeneti csatornához létrehozhatunk egy $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ alakú kernel tenzort. Összefűzzük ezeket a kimeneti csatorna dimenzióban, így a konvolúciós kernel alakja $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ lesz. A keresztkorreláció műveletekben minden kimeneti csatorna eredménye az adott kimeneti csatornának megfelelő konvolúciós kernelből kerül kiszámításra, és a bemeneti tenzor összes csatornájából vesz bemenetet.

Implementálunk egy keresztkorreláció függvényt [**több csatorna kimenetének kiszámítására**] az alábbiak szerint.

```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # Végigiterálunk K 0. dimenzióján, és minden lépésben keresztkorrelációt
    # végzünk az X bemeneten. Az összes eredményt egymásra halmozzuk
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Egy triviális, három kimeneti csatornájú konvolúciós kernelt építünk a `K` kernel tenzorát `K+1`-gyel és `K+2`-vel összefűzve.

```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Az alábbiakban keresztkorreláció műveleteket végzünk az `X` bemeneti tenzoron a `K` kernel tenzorral. A kimenet most három csatornát tartalmaz. Az első csatorna eredménye megegyezik az előző `X` bemeneti tenzor és a több bemeneti csatornával, egyetlen kimeneti csatornával rendelkező kernel eredményével.

```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$-es konvolúciós réteg
:label:`subsec_1x1`

Első pillantásra egy [**$1 \times 1$-es konvolúció**], azaz $k_\textrm{h} = k_\textrm{w} = 1$, nem tűnik különösen értelmesnek. Végül is a konvolúció szomszédos pixeleket korrelál. Egy $1 \times 1$-es konvolúció nyilvánvalóan nem teszi ezt. Mégis népszerű műveletek, amelyeket néha bonyolult mély hálózatok tervezésébe foglalnak :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`. Nézzük meg részletesen, hogy valójában mit tesz.

Mivel a minimális ablakot használja, a $1\times 1$-es konvolúció elveszíti a nagyobb konvolúciós rétegek azon képességét, hogy a magassági és szélességi dimenziókban szomszédos elemek kölcsönhatásaiból álló mintákat ismerjenek fel. A $1\times 1$-es konvolúció egyetlen számítása a csatorna dimenzióban történik.

A :numref:`fig_conv_1x1` a $1\times 1$-es konvolúciós kernelt alkalmazó keresztkorreláció számítást mutatja 3 bemeneti csatornával és 2 kimeneti csatornával. Vegyük figyelembe, hogy a bemeneteknek és kimeneteknek azonos magassága és szélessége van. A kimenet minden eleme a bemeneti kép *azonos pozícióján* lévő elemek lineáris kombinációjából adódik. A $1\times 1$-es konvolúciós rétegre úgy gondolhatunk, mint egy minden egyes pixel-helyen alkalmazott teljesen összekötött rétegre, amely a $c_\textrm{i}$ megfelelő bemeneti értéket $c_\textrm{o}$ kimeneti értékké alakítja. Mivel ez még mindig konvolúciós réteg, a súlyok pixel-helyek között meg vannak osztva. Így a $1\times 1$-es konvolúciós réteghez $c_\textrm{o}\times c_\textrm{i}$ súly szükséges (plusz az eltolás). Vegyük figyelembe azt is, hogy a konvolúciós rétegeket általában nemlinearitások követik. Ez biztosítja, hogy a $1 \times 1$-es konvolúciók nem lehet egyszerűen egyéb konvolúciókba beolvasztani.

![A keresztkorreláció számítás a $1\times 1$-es konvolúciós kernelt alkalmazza három bemeneti csatornával és két kimeneti csatornával. A bemeneti és kimeneti azonos magasságú és szélességű.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Ellenőrizzük, hogy ez működik-e a gyakorlatban: implementálunk egy $1 \times 1$-es konvolúciót egy teljesen összekötött réteg segítségével. Egyetlen módosítást kell elvégezni az adatok alakján a mátrixszorzás előtt és után.

```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Mátrixszorzás a teljesen összekötött rétegben
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

$1\times 1$-es konvolúciók elvégzésekor a fenti függvény ekvivalens a korábban implementált `corr2d_multi_in_out` keresztkorreláció függvénnyel. Ellenőrizzük ezt néhány mintaadattal.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (3, 3, 3)) + 0 * 1
K = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 3, 1, 1)) + 0 * 1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Vita

A csatornák lehetővé teszik mindkét világ legjobbjainak kombinálását: az MLP-knek, amelyek jelentős nemlinearitást tesznek lehetővé, és a konvolúcióknak, amelyek a jellemzők *lokalizált* elemzését teszik lehetővé. Különösen a csatornák lehetővé teszik a CNN számára, hogy egyszerre több jellemzővel, például él- és alakdetektorokkal logikusan következtessen. Gyakorlati kompromisszumot is kínálnak az eltolás-invarianciából és lokalitásból eredő drasztikus paramétercsökkentés, valamint a számítógépes látásban szükséges kifejező és sokszínű modellek igénye között.

Megjegyezzük azonban, hogy ez a rugalmasság árat jelent. Egy $(h \times w)$ méretű kép esetén egy $k \times k$-os konvolúció számítási költsége $\mathcal{O}(h \cdot w \cdot k^2)$. $c_\textrm{i}$ és $c_\textrm{o}$ bemeneti és kimeneti csatornák esetén ez $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$-ra nő. Egy $256 \times 256$ pixeles kép esetén $5 \times 5$-ös kernellel és $128$ bemeneti és kimeneti csatornával ez több mint 53 milliárd műveletet jelent (a szorzásokat és összeadásokat külön számolva). Később hatékony stratégiákkal találkozunk a költségek csökkentésére, például azzal, hogy a csatornánkénti műveleteket blokk-átlóssá tesszük, ami ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017` architektúrákhoz vezet.

## Feladatok

1. Tegyük fel, hogy van két $k_1$ és $k_2$ méretű konvolúciós kernelünk (köztük nincs nemlinearitás).
    1. Bizonyítsuk be, hogy a művelet eredménye egyetlen konvolúcióval kifejezhető.
    1. Mekkora az ekvivalens egyetlen konvolúció dimenzionalitása?
    1. Igaz-e az ellenkezője, azaz mindig felbontható-e egy konvolúció két kisebbre?
1. Feltételezve egy $c_\textrm{i}\times h\times w$ alakú bemenetet és egy $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ alakú konvolúciós kernelt, $(p_\textrm{h}, p_\textrm{w})$ párnázással és $(s_\textrm{h}, s_\textrm{w})$ lépésközzel.
    1. Mekkora az előreterjesztés számítási költsége (szorzások és összeadások)?
    1. Mekkora a memóriaigény?
    1. Mekkora a memóriaigény a visszaterjesztési számításhoz?
    1. Mekkora a visszaterjesztés számítási költsége?
1. Hányszorosára növekszik a számítások száma, ha mind a bemeneti csatornák számát $c_\textrm{i}$, mind a kimeneti csatornák számát $c_\textrm{o}$ megduplázzuk? Mi történik, ha megduplázzuk a párnázást?
1. Pontosan ugyanazok-e az `Y1` és `Y2` változók ennek a résznek az utolsó példájában? Miért?
1. Fejezzük ki a konvolúciókat mátrixszorzásként, még akkor is, ha a konvolúciós ablak nem $1 \times 1$.
1. Feladatunk egy $k \times k$-os kernellel végzett gyors konvolúció implementálása. Az egyik algoritmjelölt a forrást vízszintesen pásztázza, $k$-szélességű sávot olvasva, és az $1$-szélességű kimeneti sávot egyszerre egy értékkel számítja. Az alternatíva egy $k + \Delta$ szélességű sáv olvasása és egy $\Delta$-szélességű kimeneti sáv kiszámítása. Miért előnyösebb az utóbbi? Van-e korlátja, hogy mekkora $\Delta$-t válasszunk?
1. Tegyük fel, hogy van egy $c \times c$ mátrixunk.
    1. Mennyivel gyorsabb a szorzás blokk-átlós mátrixszal, ha a mátrix $b$ blokkra van bontva?
    1. Mi a $b$ blokk hátrányai? Hogyan lehetne legalább részben orvosolni?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17998)
:end_tab:
