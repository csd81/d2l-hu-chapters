```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Lekérdezések, kulcsok és értékek
:label:`sec_queries-keys-values`

Eddig minden általunk áttekintett hálózat döntően arra támaszkodott, hogy a bemenet jól meghatározott méretű legyen. Például az ImageNet képei $224 \times 224$ pixel méretűek, és a CNN-ek kifejezetten erre a méretre vannak hangolva. Még a természetes nyelvfeldolgozásban is jól meghatározott és rögzített az RNN-ek bemeneti mérete. A változó méretet egyszerre egy token szekvenciális feldolgozásával vagy speciálisan tervezett konvolúciós kernelekkel kezelik :cite:`Kalchbrenner.Grefenstette.Blunsom.2014`. Ez a megközelítés jelentős problémákhoz vezethet, amikor a bemenet valóban változó méretű és változó információtartalommal rendelkezik, mint például a :numref:`sec_seq2seq`-ben a szöveg átalakításakor :cite:`Sutskever.Vinyals.Le.2014`. Különösen hosszú szekvenciák esetén meglehetősen nehézzé válik nyomon követni mindazt, amit a hálózat már generált vagy akár látott. Még az olyan explicit nyomkövetési heurisztikák, mint amilyeneket :citet:`yang2016neural` javasolt, is csak korlátozott előnnyel bírnak.

Hasonlítsuk ezt az adatbázisokhoz. Legegyszerűbb formájukban kulcsok ($k$) és értékek ($v$) gyűjteményei. Például a $\mathcal{D}$ adatbázisunk állhat a következő párokból: \{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")\}, ahol a kulcs a vezetéknév és az érték az utónév. Műveleteket hajthatunk végre a $\mathcal{D}$-n, például a „Li" pontos lekérdezésével ($q$), amely a „Mu" értéket adná vissza. Ha a ("Li", "Mu") bejegyzés nem szerepelne $\mathcal{D}$-ben, nem lenne érvényes válasz. Ha hozzávetőleges egyezéseket is megengedünk, akkor ("Lipton", "Zachary") kerülne visszaadásra. Ez az egyszerű és triviálisnak tűnő példa mégis számos hasznos dolgot tanít nekünk:

* Olyan $q$ lekérdezéseket tervezhetünk, amelyek ($k$,$v$) párokra működnek oly módon, hogy érvényesek az adatbázis méretétől függetlenül.
* Ugyanaz a lekérdezés különböző válaszokat kaphat az adatbázis tartalmától függően.
* A nagy állapottérre (az adatbázisra) vonatkozó műveletek végrehajtásához szükséges „kód" meglehetősen egyszerű lehet (pl. pontos egyezés, hozzávetőleges egyezés, top-$k$).
* Nincs szükség az adatbázis tömörítésére vagy egyszerűsítésére ahhoz, hogy a műveletek hatékonyak legyenek.

Nyilván nem vezettük volna be itt ezt az egyszerű adatbázist, ha nem a mélytanulás magyarázata érdekében tettük volna. Valóban, ez elvezet az elmúlt évtizedben a mélytanulásban bevezetett egyik legizgalmasabb fogalomhoz: a *figyelemmechanizmushoz* :cite:`Bahdanau.Cho.Bengio.2014`. A gépi fordításra való alkalmazásának részleteit később tárgyaljuk. Egyelőre csak vegyük figyelembe a következőket: jelöljük $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$ a *kulcsok* és *értékek* $m$ párjából álló adatbázist. Továbbá jelöljük $\mathbf{q}$-val a *lekérdezést*. Ekkor definiálhatjuk $\mathcal{D}$ feletti *figyelmet* a következőképpen:

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

ahol $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ ($i = 1, \ldots, m$) skaláris figyelemsúlyok. Magát a műveletet általában *figyelempooling*-nak (*attention pooling*) nevezik. A *figyelem* elnevezés abból a tényből ered, hogy a művelet különös figyelmet fordít azokra a tagokra, amelyekre az $\alpha$ súly jelentős (azaz nagy). Mint ilyen, a $\mathcal{D}$ feletti figyelem az adatbázisban tárolt értékek lineáris kombinációját állítja elő. Valójában ez tartalmazza a fenti példát speciális esetként, ahol az összes súly nulla, kivéve egyet. Számos speciális eset van:

* A $\alpha(\mathbf{q}, \mathbf{k}_i)$ súlyok nemnegatívak. Ebben az esetben a figyelemmechanizmus kimenete a $\mathbf{v}_i$ értékek által kifeszített konvex kúpban van.
* A $\alpha(\mathbf{q}, \mathbf{k}_i)$ súlyok konvex kombinációt alkotnak, azaz $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$ és $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$ minden $i$-re. Ez a mélytanulás leggyakoribb beállítása.
* Pontosan egy súly egyenlő $1$-gyel, a $\alpha(\mathbf{q}, \mathbf{k}_i)$ értékek közül, míg az összes többi $0$. Ez a hagyományos adatbázis-lekérdezéshez hasonló.
* Minden súly egyenlő, azaz $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$ minden $i$-re. Ez az egész adatbázis átlagolásának felel meg, amit mélytanulásban average pooling-nak is neveznek.

A súlyok összegének $1$-re való normalizálásának általános stratégiája az, hogy normalizáljuk őket:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Különösen a súlyok nemnegatívságának biztosítására hatványozáshoz folyamodhatunk. Ez azt jelenti, hogy most már *bármilyen* $a(\mathbf{q}, \mathbf{k})$ függvényt kiválaszthatunk, majd alkalmazhatjuk rá a multinomiális modellekhez használt softmax műveletet:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}. $$
:eqlabel:`eq_softmax_attention`

Ez a művelet minden mélytanulási keretrendszerben könnyen elérhető. Differenciálható, és a gradiense soha nem tűnik el, amelyek mind kívánatos tulajdonságok egy modellben. Megjegyzendő azonban, hogy a fent bevezetett figyelemmechanizmus nem az egyetlen lehetőség. Például tervezhetünk nem differenciálható figyelemmodellt, amelyet megerősítéses tanulási módszerekkel lehet tanítani :cite:`Mnih.Heess.Graves.ea.2014`. Ahogyan várható, egy ilyen modell tanítása meglehetősen összetett. Következésképpen a modern figyelemkutatás nagy része a :numref:`fig_qkv`-ban vázolt keretet követi. Ezért az exponálásunkat erre a differenciálható mechanizmusok családjára összpontosítjuk.

![A figyelemmechanizmus lineáris kombinációt számít a $\mathbf{v}_\mathit{i}$ értékekre figyelempooling segítségével, ahol a súlyok a $\mathbf{q}$ lekérdezés és a $\mathbf{k}_\mathit{i}$ kulcsok közötti kompatibilitás alapján kerülnek levezetésre.](../img/qkv.svg)
:label:`fig_qkv`

Meglehetősen figyelemre méltó, hogy a kulcsok és értékek halmazán végrehajtott tényleges „kód", vagyis a lekérdezés elég tömör lehet, annak ellenére, hogy a kezelendő tér jelentős. Ez kívánatos tulajdonság egy hálózati réteg számára, mivel nem igényel túl sok tanítható paramétert. Ugyanilyen kényelmes az a tény, hogy a figyelem tetszőlegesen nagy adatbázisokon is működhet anélkül, hogy meg kellene változtatni a figyelempooling-művelet végrehajtásának módját.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Vizualizáció

A figyelemmechanizmus egyik előnye, hogy meglehetősen szemléletes lehet, különösen akkor, ha a súlyok nemnegatívak és összegük $1$. Ebben az esetben a nagy súlyokat úgy *értelmezhetjük*, mint amelyek segítségével a modell releváns komponenseket választ ki. Bár ez jó intuíció, fontos emlékezni arra, hogy ez csupán *intuíció*. Mindenesetre érdemes lehet vizualizálni a figyelem hatását az adott kulcshalmazra, különféle lekérdezések alkalmazásakor. Ez a függvény később jól jön majd.

Ezért definiáljuk a `show_heatmaps` függvényt. Figyeljük meg, hogy ez nem egy (figyelemsúly-) mátrixot vesz bemenetként, hanem egy négy tengellyel rendelkező tenzort, amely lehetővé teszi különböző lekérdezések és súlyok tömbjét. Következésképpen a `matrices` bemeneti alakja (a megjelenítési sorok száma, a megjelenítési oszlopok száma, a lekérdezések száma, a kulcsok száma). Ez majd jól jön, amikor vizualizálni akarjuk a Transformerek tervezéséhez szükséges működést.

```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Mátrixok hőtérképeinek megjelenítése."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if tab.selected('jax'):
                pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Gyors ellenőrzésképpen vizualizáljuk az egységmátrixot, amely azt az esetet jelöli, amikor a figyelemsúly csak akkor $1$, ha a lekérdezés és a kulcs azonos.

```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Kulcsok', ylabel='Lekérdezések')
```

## Összefoglalás

A figyelemmechanizmus lehetővé teszi számunkra, hogy sok (kulcs, érték) párból összesítsük az adatokat. Eddig a tárgyalásunk meglehetősen elvont volt, egyszerűen leírva az adatok összesítésének módját. Még nem magyaráztuk el, honnan kerülhetnek elő ezek a rejtélyes lekérdezések, kulcsok és értékek. Néhány intuíció segíthet itt: például regressziós esetben a lekérdezés megfelelhet annak a helynek, ahol a regressziót el kell végezni. A kulcsok azok a helyek, ahol a múltbeli adatokat megfigyelték, és az értékek maguk a (regressziós) értékek. Ez az úgynevezett Nadaraya–Watson-becslő :cite:`Nadaraya.1964,Watson.1964`, amelyet a következő szakaszban fogunk tanulmányozni.

Tervezés szerint a figyelemmechanizmus *differenciálható* vezérlési eszközt biztosít, amellyel egy neurális hálózat elemeket választhat ki egy halmazból, és felépíthet egy kapcsolódó súlyozott összegzést a reprezentációk felett.

## Feladatok

1. Tételezzük fel, hogy a klasszikus adatbázisokban használt hozzávetőleges (kulcs, lekérdezés) egyezéseket szeretnénk újraimplementálni. Melyik figyelemfüggvényt választanád?
1. Tételezzük fel, hogy a figyelemfüggvényt $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ adja meg, és $\mathbf{k}_i = \mathbf{v}_i$ teljesül $i = 1, \ldots, m$-re. Jelölje $p(\mathbf{k}_i; \mathbf{q})$ a kulcsok valószínűségi eloszlását a :eqref:`eq_softmax_attention`-ban lévő softmax normalizáció használatakor. Bizonyítsd be, hogy $\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$.
1. Tervezz differenciálható keresőmotort a figyelemmechanizmus segítségével.
1. Tekintsd át a Squeeze and Excitation Networks :cite:`Hu.Shen.Sun.2018` tervezését, és értelmezd azokat a figyelemmechanizmus szemüvegén keresztül.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18024)
:end_tab:
