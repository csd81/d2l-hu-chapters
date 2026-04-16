```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# Mi a hiperparaméter-optimalizálás?
:label:`sec_what_is_hpo`

Ahogy az előző fejezetekben láttuk, a mély neurális hálózatok nagyszámú paramétert vagy súlyt tartalmaznak, amelyeket a tanítás során tanulnak meg. Ezeken felül minden neurális hálózatnak vannak további *hiperparaméterei*, amelyeket a felhasználónak kell beállítania. Például ahhoz, hogy a sztochasztikus gradienscsökkentés a tanítási veszteség egy lokális optimumához konvergáljon (lásd :numref:`chap_optimization`), be kell állítanunk a tanulási rátát és a batch méretet. A tanítási adathalmazon való túlillesztés elkerülése érdekében regularizációs paramétereket kell megadnunk, mint például a súlycsökkentés (lásd :numref:`sec_weight_decay`) vagy a dropout (lásd :numref:`sec_dropout`). A modell kapacitását és induktív torzítását a rétegek számával, illetve a rétegenként lévő egységek vagy szűrők számával (azaz a tényleges súlyok számával) határozhatjuk meg.

Sajnos ezeket a hiperparamétereket nem tudjuk egyszerűen a tanítási veszteség minimalizálásával beállítani, mivel ez túlillesztéshez vezetne a tanítási adatokon. Például a regularizációs paraméterek, mint a dropout vagy a súlycsökkentés, nullára állítása kis tanítási veszteséget eredményez, de ronthatja az általánosítási teljesítményt.

![A gépi tanulás tipikus munkafolyamata, amelynek során a modellt többször tanítjuk különböző hiperparaméterekkel.](../img/ml_workflow.svg)
:label:`ml_workflow`

Automatizálás nélkül a hiperparamétereket manuálisan kell beállítani próba-és-hiba módszerrel, ami a gépi tanulási munkafolyamatok időigényes és nehéz részét képezi. Például egy ResNet (lásd :numref:`sec_resnet`) tanítása CIFAR-10-en több mint 2 órát vesz igénybe egy Amazon Elastic Cloud Compute (EC2) `g4dn.xlarge` példányon. Még ha csak tíz hiperparaméter-konfigurációt próbálunk ki egymás után, ez már nagyjából egy teljes napot vesz igénybe. Ráadásul a hiperparaméterek általában nem vihetők át közvetlenül különböző architektúrák és adathalmazok között :cite:`feurer-arxiv22,wistuba-ml18,bardenet-icml13a`, és minden új feladathoz újra kell optimalizálni őket. Emellett a legtöbb hiperparaméterhez nincsenek ökölszabályok, és az ésszerű értékek megtalálásához szakértői tudás szükséges.

A *hiperparaméter-optimalizáló (HPO)* algoritmusok célja, hogy elvszerű és automatizált módon kezeljék ezt a problémát :cite:`feurer-automlbook18a`, globális optimalizálási feladatként fogalmazva meg azt. Az alapértelmezett célkitűzés a félretett validációs adathalmazon mért hiba, de elvben bármilyen más üzleti metrika is lehet. Kombinálható vagy korlátozható másodlagos célkitűzésekkel, mint például a tanítási idő, az inferencia ideje vagy a modell összetettsége.

Az utóbbi időben a hiperparaméter-optimalizálást kiterjesztették a *neurális architektúrakeresésre (NAS)* :cite:`elsken-arxiv18a,wistuba-arxiv19`, amelynek célja teljesen új neurális hálózati architektúrák megtalálása. A klasszikus HPO-hoz képest a NAS még számításigényesebb, és további erőfeszítéseket igényel a gyakorlati megvalósíthatóság érdekében. Mind a HPO, mind a NAS az AutoML :cite:`hutter-book19a` részterületeinek tekinthető, amelynek célja a teljes gépi tanulási folyamat automatizálása.

Ebben a szakaszban bevezetjük a HPO-t, és megmutatjuk, hogyan találhatjuk meg automatikusan a :numref:`sec_softmax_concise`-ban bemutatott logisztikus regressziós példa legjobb hiperparamétereit.

##  Az optimalizálási feladat
:label:`sec_definition_hpo`

Egy egyszerű játékpéldával kezdünk: a :numref:`sec_softmax_concise`-beli `SoftmaxRegression` többosztályos logisztikus regressziós modell tanulási rátáját keressük, hogy minimalizáljuk a validációs hibát a Fashion MNIST adathalmazon. Bár más hiperparaméterek, mint a batch méret vagy az epochok száma is érdemes hangolni, az egyszerűség kedvéért csak a tanulási rátára koncentrálunk.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from scipy import stats
```

Mielőtt futtathatnánk a HPO-t, két összetevőt kell meghatároznunk: a célfüggvényt és a konfigurációs teret.

### A célfüggvény

Egy tanulóalgoritmus teljesítménye egy $f: \mathcal{X} \rightarrow \mathbb{R}$ függvényként tekinthető, amely a hiperparaméter-térből $\mathbf{x} \in \mathcal{X}$ a validációs veszteségre képez le. Az $f(\mathbf{x})$ minden kiértékelésekor el kell tanítanunk és validálnunk a gépi tanulási modellünket, ami nagy adathalmazokon tanított mély neurális hálózatok esetén időigényes és számításigényes lehet. Az $f(\mathbf{x})$ kritériumunk alapján célunk megtalálni a $\mathbf{x}_{\star} \in \mathrm{argmin}_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ értéket.

Nincs egyszerű mód az $f$ gradiensének kiszámítására $\mathbf{x}$ szerint, mivel ehhez a gradienst a teljes tanítási folyamaton kellene visszavezetni. Bár léteznek friss kutatások :cite:`maclaurin-icml15,franceschi-icml17a`, amelyek közelítő „hipergradiensekkel" próbálják vezérelni a HPO-t, egyik megközelítés sem versenyképes még a legkorszerűbb módszerekkel, ezért itt nem tárgyaljuk őket. Ráadásul az $f$ kiértékelésének számítási terhe megköveteli, hogy a HPO-algoritmusok minél kevesebb mintával közelítsék meg a globális optimumot.

A neurális hálózatok tanítása sztochasztikus (pl. a súlyok véletlenszerűen inicializálódnak, a mini-batch-ek véletlenszerűen kerülnek kiválasztásra), ezért megfigyeléseink zajosak lesznek: $y \sim f(\mathbf{x}) + \epsilon$, ahol általában azt feltételezzük, hogy az $\epsilon \sim N(0, \sigma)$ megfigyelési zaj Gauss-eloszlást követ.

Mindezekkel a kihívásokkal szembesülve általában arra törekszünk, hogy gyorsan azonosítsunk egy kis halmaz jól teljesítő hiperparaméter-konfigurációt, ahelyett hogy pontosan elérnénk a globális optimumot. A legtöbb neurális hálózati modell nagy számítási igénye miatt azonban még ez is napokig vagy hetekig tarthat. A :numref:`sec_mf_hpo` részben megvizsgáljuk, hogyan gyorsíthatjuk fel az optimalizálási folyamatot akár a keresés elosztásával, akár a célfüggvény olcsóbban kiértékelhető közelítéseinek alkalmazásával.

Kezdjük egy módszerrel, amely kiszámítja a modell validációs hibáját.

```{.python .input  n=8}
%%tab pytorch
class HPOTrainer(d2l.Trainer):  #@save
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_idx = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_idx += 1
        return 1 -  accuracy / val_batch_idx
```

A validációs hibát a `learning_rate`-ből álló `config` hiperparaméter-konfiguráció szerint optimalizáljuk. Minden kiértékeléskor `max_epochs` epochig tanítjuk a modellt, majd kiszámítjuk és visszaadjuk a validációs hibáját:

```{.python .input  n=5}
%%tab pytorch
def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config["learning_rate"]
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return d2l.numpy(trainer.validation_error())
```

### A konfigurációs tér
:label:`sec_intro_config_spaces`

Az $f(\mathbf{x})$ célfüggvény mellett meg kell határoznunk azt a megvalósítható halmazt $\mathbf{x} \in \mathcal{X}$ is, amelyen optimalizálunk; ezt *konfigurációs térnek* vagy *keresési térnek* nevezzük. A logisztikus regressziós példánkhoz a következőt fogjuk használni:

```{.python .input  n=6}
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

Itt a SciPy `loguniform` objektumát használjuk, amely egy logaritmikus skálán -4 és -1 közötti egyenletes eloszlást képvisel. Ez az objektum lehetővé teszi, hogy véletlen változókat mintavételezzünk ebből az eloszlásból.

Minden hiperparaméternek van adattípusa, például `float` a `learning_rate` esetén, valamint egy zárt, korlátozott tartománya (azaz alsó és felső határ). Általában prior eloszlást rendelünk minden hiperparaméterhez (pl. egyenletes vagy log-egyenletes), amelyből mintavételezünk. Egyes pozitív paramétereket, mint a `learning_rate`, logaritmikus skálán célszerű ábrázolni, mivel az optimális értékek több nagyságrenddel is eltérhetnek, míg mások, mint a momentum, lineáris skálán értelmezhetők.

Az alábbiakban egy egyszerű példát mutatunk egy többrétegű perceptron tipikus hiperparamétereiből álló konfigurációs térre, a típusokkal és a szokásos tartományokkal együtt.

: Példa többrétegű perceptron konfigurációs terére
:label:`tab_example_configspace`

| Név                    | Típus       | Hiperparaméter tartomány           | log-skála |
| :----:                 | :----:      |:----------------------------------:|:---------:|
| tanulási ráta          | float       |      $[10^{-6},10^{-1}]$           |    igen   |
| batch méret            | integer     |           $[8,256]$                |    igen   |
| momentum               | float       |           $[0,0.99]$               |    nem    |
| aktivációs függvény    | kategorikus | $\{\textrm{tanh}, \textrm{relu}\}$ |     -     |
| egységek száma         | integer     |          $[32, 1024]$              |    igen   |
| rétegek száma          | integer     |            $[1, 6]$                |    nem    |



Általánosságban a konfigurációs tér $\mathcal{X}$ szerkezete összetett lehet, és jelentősen eltérhet $\mathbb{R}^d$-től. A gyakorlatban egyes hiperparaméterek más hiperparaméterek értékétől függhetnek. Például tegyük fel, hogy egy többrétegű perceptron rétegszámát, illetve rétegenkénti egységszámát szeretnénk hangolni. Az $l\textrm{-edik}$ réteg egységeinek száma csak akkor releváns, ha a hálózatnak legalább $l+1$ rétege van. Ezek a haladó HPO-problémák meghaladják e fejezet kereteit; az érdeklődő olvasót :cite:`hutter-lion11a,jenatton-icml17a,baptista-icml18a` munkákhoz irányítjuk.

A konfigurációs tér kulcsfontosságú szerepet tölt be a hiperparaméter-optimalizálásban, mivel egyetlen algoritmus sem találhat olyat, ami nincs benne a konfigurációs térben. Másfelől, ha a tartományok túl nagyok, a jól teljesítő konfigurációk megtalálásához szükséges számítási keret megfizethetetlenné válhat.

## Véletlen keresés
:label:`sec_rs`

A *véletlen keresés* az első hiperparaméter-optimalizáló algoritmus, amelyet megvizsgálunk. A véletlen keresés alapgondolata az, hogy egymástól függetlenül mintavételezünk a konfigurációs térből, amíg egy előre meghatározott keretet (pl. a maximális iterációszámot) ki nem merítjük, majd visszaadjuk a legjobb megfigyelt konfigurációt. Minden kiértékelés párhuzamosan, egymástól függetlenül végrehajtható (lásd :numref:`sec_rs_async`), de az egyszerűség kedvéért itt szekvenciális ciklust használunk.

```{.python .input  n=7}
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
    learning_rate = config_space["learning_rate"].rvs()
    print(f"Trial {i}: learning_rate = {learning_rate}")
    y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
    print(f"    validation_error = {y}")
    values.append(learning_rate)
    errors.append(y)
```

A legjobb tanulási ráta egyszerűen az, amelyhez a legkisebb validációs hiba tartozik.

```{.python .input  n=7}
best_idx = np.argmin(errors)
print(f"optimal learning rate = {values[best_idx]}")
```

Egyszerűsége és általánossága miatt a véletlen keresés az egyik leggyakrabban használt HPO-algoritmus. Nem igényel bonyolult implementációt, és bármilyen konfigurációs térre alkalmazható, amennyiben minden hiperparaméterhez meg tudunk adni valamilyen valószínűségi eloszlást.

A véletlen keresésnek sajnos vannak hiányosságai is. Egyrészt nem igazítja a mintavételezési eloszlást az eddig összegyűjtött korábbi megfigyelések alapján. Ezért egy rosszul teljesítő konfigurációt ugyanolyan valószínűséggel mintavételez, mint egy jobban teljesítőt. Másrészt minden konfigurációra ugyanannyi erőforrást fordít, még akkor is, ha néhány gyenge kezdeti teljesítményt mutat, és kisebb valószínűséggel múlja felül a korábban látott konfigurációkat.

A következő szakaszokban mintahatékonyabb hiperparaméter-optimalizáló algoritmusokat vizsgálunk meg, amelyek modell segítségével irányítják a keresést, leküzdve a véletlen keresés hiányosságait. Megvizsgálunk olyan algoritmusokat is, amelyek automatikusan leállítják a gyengén teljesítő konfigurációk kiértékelési folyamatát, hogy felgyorsítsák az optimalizálást.

## Összefoglalás

Ebben a szakaszban bevezettük a hiperparaméter-optimalizálást (HPO), és megmutattuk, hogyan fogalmazható meg globális optimalizálási feladatként egy konfigurációs tér és egy célfüggvény meghatározásával. Megvalósítottuk az első HPO-algoritmusunkat is, a véletlen keresést, és alkalmaztuk egy egyszerű softmax osztályozási problémán.

Bár a véletlen keresés igen egyszerű, jobb alternatíva a rácsos keresésnél, amely egyszerűen egy rögzített hiperparaméter-halmazt értékel ki. A véletlen keresés némileg enyhíti az átok dimenzióit :cite:`bellman-science66`, és sokkal hatékonyabb lehet a rácsos keresésnél, ha a kritérium döntően a hiperparaméterek egy kis részhalmazától függ.

## Feladatok

1. Ebben a fejezetben egy modell validációs hibáját optimalizáljuk, miután egy diszjunkt tanítóhalmazon tanítottuk. Az egyszerűség kedvéért a kódunk a `Trainer.val_dataloader`-t használja, amely egy `FashionMNIST.val` köré épített betöltőre mutat.
    1. Győzd meg magad (a kód megtekintésével), hogy ez azt jelenti: az eredeti FashionMNIST tanítóhalmazt (60000 példa) használjuk tanításhoz, az eredeti *teszthalmazt* (10000 példa) pedig validációhoz.
    2. Miért lehet problémás ez a gyakorlat? Útmutatás: Olvasd újra :numref:`sec_generalization_basics`-t, különösen a *modellkiválasztásról* szóló részt.
    3. Mit kellett volna helyette tennünk?
2. Fentebb megjegyeztük, hogy a gradienscsökkentésen alapuló hiperparaméter-optimalizálás igen nehéz. Vegyünk egy kis problémát: egy kétrétegű perceptron tanítása a FashionMNIST adathalmazon (:numref:`sec_mlp-implementation`) 256-os batch mérettel. Az SGD tanulási rátáját szeretnénk hangolni, hogy minimalizáljuk a validációs metrikát egy epoch tanítás után.
    1. Miért nem használhatjuk erre a célra a validációs *hibát*? Milyen metrikát használnál a validációs halmazon?
    2. Vázold fel (nagyjából) a validációs metrika számítási gráfját egy epoch tanítás után. Feltételezheted, hogy a kezdeti súlyok és hiperparaméterek (pl. tanulási ráta) ennek a gráfnak bemeneti csomópontjai. Útmutatás: Olvasd újra a számítási gráfokról szóló részt :numref:`sec_backprop`-ban.
    3. Becsüld meg nagyjából, hány lebegőpontos értéket kell tárolnod egy előrepassz során ezen a gráfon. Útmutatás: A FashionMNIST-nek 60000 esete van. Feltételezd, hogy a szükséges memóriát az egyes rétegek utáni aktivációk dominálják, és keresd meg a rétegszélességeket :numref:`sec_mlp-implementation`-ban.
    5. A szükséges számítási és tárolási kapacitáson túl milyen egyéb problémákba ütközne a gradient alapú hiperparaméter-optimalizálás? Útmutatás: Olvasd újra az elhaló és felrobbanó gradiensekről szóló részt :numref:`sec_numerical_stability`-ban.
    6. *Haladó*: Olvasd el :cite:`maclaurin-icml15` munkát egy elegáns (bár még nem egészen praktikus) gradient alapú HPO-megközelítésért.
3. A rácsos keresés egy másik HPO-alapvonal, ahol minden hiperparaméterhez egyenletes rácsot definiálunk, majd a (kombinatorikus) Descartes-szorzaton iterálunk át konfigurációk javaslásához.
    1. Fentebb megjegyeztük, hogy a véletlen keresés sok hiperparaméter esetén jóval hatékonyabb lehet a rácsos keresésnél, ha a kritérium döntően a hiperparaméterek egy kis részhalmazától függ. Miért van ez? Útmutatás: Olvasd el :cite:`bergstra2011algorithms` munkát.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12090)
:end_tab:
