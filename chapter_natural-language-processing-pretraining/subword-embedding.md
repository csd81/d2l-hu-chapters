# Részszó-beágyazás
:label:`sec_fasttext`

Az angolban
az olyan szavak, mint
a „helps", a „helped" és a „helping"
a „help" szó ragozott alakjai.
A „dog" és a „dogs" közötti kapcsolat
megegyezik
a „cat" és a „cats" közötti kapcsolattal,
és a „boy" és a „boyfriend" közötti összefüggés
párhuzamos
a „girl" és a „girlfriend" közöttivel.
Más nyelvekben,
például a franciában és a spanyolban,
számos igének több mint 40 ragozott alakja van,
míg a finnben
egy főnévnek akár 15 esete is lehet.
A nyelvészetben
a morfológia a szóképzést és a szavak kapcsolatait vizsgálja.
A szavak belső szerkezetét azonban
sem a word2vec,
sem a GloVe nem vizsgálta.

## A fastText modell

Idézzük fel, hogyan reprezentálják a szavakat a word2vec-ben.
Mind a skip-gram modellben,
mind a folytonos szózsák-modellben (continuous bag-of-words)
ugyanazon szó különböző ragozott alakjait
közvetlenül különböző vektorok reprezentálják,
megosztott paraméterek nélkül.
A morfológiai információ felhasználásához
a *fastText* modell
egy *részszó-beágyazási* megközelítést javasolt,
amelyben a részszó egy karakter-$n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017`.
Ahelyett, hogy szószintű vektorreprezentációkat tanulna,
a fastText a részszó-szintű skip-gram-ként fogható fel,
amelyben minden *középső szót*
a részszó-vektorainak összege reprezentál.

Mutassuk be, hogyan kapjuk meg
a fastText-ben minden középső szó részszavait
a „where" szó példáján.
Először a szó elejéhez és végéhez
adjuk hozzá a speciális „&lt;" és „&gt;" karaktereket,
hogy az előtagokat és utótagokat meg lehessen különböztetni a többi részszótól.
Ezután vonjuk ki a karakter-$n$-gramokat a szóból.
Például $n=3$ esetén
az összes 3 hosszúságú részszót kapjuk: „&lt;wh", „whe", „her", „ere", „re&gt;", és a speciális „&lt;where&gt;" részszót.


A fastText-ben bármely $w$ szóra
jelöljük $\mathcal{G}_w$-vel
a 3 és 6 közötti hosszúságú összes részszava
és a speciális részszava unióját.
A szótár
az összes szó részszavainak uniója.
Ha $\mathbf{z}_g$
a szótárban szereplő $g$ részszó vektora,
akkor a skip-gram modellben
a $w$ szó $\mathbf{v}_w$ középső szó vektora
a részszó-vektorok összege:

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

A fastText többi része megegyezik a skip-gram modellével. A skip-gram modellel összehasonlítva
a fastText szótára nagyobb,
ami több modellparamétert jelent.
Emellett
egy szó reprezentációjának kiszámításához
az összes részszó-vektort össze kell adni,
ami magasabb számítási komplexitást eredményez.
Azonban
a hasonló szerkezetű szavak között megosztott részszó-paramétereknek köszönhetően
a ritka szavak, sőt a szótáron kívüli szavak
is jobb vektorreprezentációt kaphatnak a fastText-ben.



## Bájt-pár kódolás (Byte Pair Encoding)
:label:`subsec_Byte_Pair_Encoding`

A fastText-ben az összes kinyert részszónak adott hosszúságúnak kell lennie, például $3$-tól $6$-ig, ezért a szótár mérete nem határozható meg előre.
Ahhoz, hogy rögzített méretű szótárban változó hosszúságú részszavak is szerepeljenek,
alkalmazhatunk egy *bájt-pár kódolás* (byte pair encoding, BPE) nevű tömörítési algoritmust
részszavak kinyerésére :cite:`Sennrich.Haddow.Birch.2015`.

A bájt-pár kódolás statisztikai elemzést végez a tanítóhalmazon,
hogy megtalálja a szón belüli leggyakoribb szimbólumokat,
például tetszőleges hosszúságú egymást követő karaktereket.
Az 1-es hosszúságú szimbólumoktól kiindulva
a bájt-pár kódolás iteratívan egyesíti az egymást követő szimbólumok
leggyakoribb párját, hogy új, hosszabb szimbólumokat hozzon létre.
Fontos, hogy hatékonysági okokból a szóhatárokon átnyúló párokat nem vesszük figyelembe.
Végeredményben az ilyen szimbólumok részszóként használhatók szavak szegmentálásához.
A bájt-pár kódolást és változatait felhasználják bemeneti reprezentációként
olyan népszerű természetes nyelvfeldolgozási előtanítási modellekben, mint a GPT-2 :cite:`Radford.Wu.Child.ea.2019` és a RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
Az alábbiakban bemutatjuk, hogyan működik a bájt-pár kódolás.

Először a szimbólumszótárat az összes angol kisbetűs karakterre, egy speciális szóvégi szimbólumra (`'_'`) és egy speciális ismeretlen szimbólumra (`'[UNK]'`) inicializáljuk.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Mivel nem vesszük figyelembe a szóhatárokon átnyúló szimbólumpárokat,
csupán egy `raw_token_freqs` szótárra van szükségünk,
amely a szavakat egy adathalmazban megfigyelt előfordulási gyakoriságukhoz rendeli.
Figyeljük meg, hogy a speciális `'_'` szimbólumot minden szóhoz hozzáfűzzük, így
könnyedén visszaállíthatjuk a szószekvenciát (pl. „a taller man")
a kimeneti szimbólumok sorozatából (pl. „a_ tall er_ man").
Mivel az összevonási folyamatot csak egyes karakterekből és speciális szimbólumokból álló szótárból indítjuk,
szóközöket szúrunk be minden egymást követő karakterpár közé az egyes szavakon belül (a `token_freqs` szótár kulcsaiban).
Vagyis a szóköz a szavakon belüli szimbólumok elválasztójaként szolgál.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Az alábbi `get_max_freq_pair` függvény
a szóban egymást követő szimbólumok leggyakoribb párját adja vissza,
ahol a szavak a bemeneti `token_freqs` szótár kulcsaiból származnak.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # A `pairs` kulcsa két egymást követő szimbólum rendezett párja
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # A `pairs` legnagyobb értékű kulcsa
```

Az egymást követő szimbólumok gyakoriságán alapuló mohó megközelítésként
a bájt-pár kódolás az alábbi `merge_symbols` függvényt használja
az egymást követő szimbólumok leggyakoribb párjának összevonásához új szimbólumok előállítása céljából.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Most iteratívan alkalmazzuk a bájt-pár kódolás algoritmusát a `token_freqs` szótár kulcsain. Az első iterációban az egymást követő szimbólumok leggyakoribb párja `'t'` és `'a'`, ezért a bájt-pár kódolás összeolvasztja őket, és létrehozza az új `'ta'` szimbólumot. A második iterációban a bájt-pár kódolás folytatódik: `'ta'` és `'l'` összevonásával újabb szimbólum, a `'tal'` jön létre.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

A bájt-pár kódolás 10 iterációja után látható, hogy a `symbols` lista most 10 további szimbólumot tartalmaz, amelyeket iteratívan vontunk össze más szimbólumokból.

```{.python .input}
#@tab all
print(symbols)
```

A `raw_token_freqs` szótár kulcsaiban megadott adathalmazban
minden szót most a bájt-pár kódolás eredményeként
a „fast_", „fast", „er_", „tall_" és „tall" részszavakra szegmentálnak.
Például a „faster_" és a „taller_" szavakat rendre „fast er_" és „tall er_" alakra bontják.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Fontos megjegyezni, hogy a bájt-pár kódolás eredménye az alkalmazott adathalmaztól függ.
Az egyik adathalmazból tanult részszavakat
egy másik adathalmaz szavainak szegmentálásához is felhasználhatjuk.
Mohó megközelítésként a következő `segment_BPE` függvény
megpróbálja a szavakat a lehető leghosszabb részszavakra bontani
a `symbols` bemeneti argumentumból.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # A tokent a symbols-ból vett lehető leghosszabb részszavakra bontjuk
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Az alábbiakban a `symbols` listában szereplő részszavakat használjuk – amelyeket a fent említett adathalmazból tanultunk –,
hogy szegmentáljuk a `tokens` által reprezentált másik adathalmazt.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Összefoglalás

* A fastText modell egy részszó-beágyazási megközelítést javasol. A word2vec skip-gram modelljére építve a középső szót a részszó-vektorainak összegeként reprezentálja.
* A bájt-pár kódolás statisztikai elemzést végez a tanítóhalmazon a szón belüli leggyakoribb szimbólumok megtalálásához. Mohó megközelítésként iteratívan egyesíti az egymást követő szimbólumok leggyakoribb párját.
* A részszó-beágyazás javíthatja a ritka szavak és a szótáron kívüli szavak reprezentációjának minőségét.

## Feladatok

1. Például az angolban körülbelül $3\times 10^8$ lehetséges $6$-gram létezik. Mi a probléma, ha túl sok a részszó? Hogyan kezelhető ez a probléma? Útmutatás: lásd a fastText-cikk 3.2. szakaszának végét :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Hogyan tervezhető részszó-beágyazási modell a folytonos szózsák-modell alapján?
1. Ahhoz, hogy $m$ méretű szótárat kapjunk, hány összevonási műveletre van szükség, ha a kezdeti szimbólumszótár mérete $n$?
1. Hogyan terjeszthető ki a bájt-pár kódolás ötlete kifejezések kinyerésére?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4587)
:end_tab:
