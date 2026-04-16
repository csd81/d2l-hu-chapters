# Személyre szabott rangsorolás ajánlórendszerekhez

Az előző szakaszokban csak az explicit visszajelzést vettük figyelembe, és a modelleket megfigyelt értékeléseken tanítottuk és teszteltük. Az ilyen módszereknek két hátránya van. Először is, a valós rendszerekben a visszajelzés többnyire nem explicit, hanem implicit, és az explicit visszajelzés összegyűjtése költségesebb lehet. Másodszor, teljesen figyelmen kívül hagyják a meg nem figyelt felhasználó-elempárokat, pedig ezek is jelezhetik a felhasználók érdeklődését. Emiatt az ilyen módszerek nem alkalmasak azokra az esetekre, amikor az értékelések hiánya nem véletlenszerű, hanem a felhasználói preferenciákból adódik. A meg nem figyelt felhasználó-elempárok valós negatív visszajelzésekből állhatnak össze, amikor a felhasználót nem érdekli az adott elem, illetve hiányzó értékekből, amikor a felhasználó később még interakcióba léphet az elemmel. Mátrixfaktorizációban és az AutoRec-ben egyszerűen elhagyjuk a meg nem figyelt párokat. Nyilvánvaló, hogy ezek a modellek nem tudnak különbséget tenni a megfigyelt és a nem megfigyelt párok között, ezért általában nem alkalmasak személyre szabott rangsorolási feladatokra.

E célból olyan ajánlási modellek váltak népszerűvé, amelyek implicit visszajelzésből rangsorolt ajánlási listákat állítanak elő. Általánosságban a személyre szabott rangsorolási modellek pontonkénti, páronkénti vagy lista-alapú megközelítésekkel optimalizálhatók. A pontonkénti megközelítések egyszerre egyetlen interakciót kezelnek, és egy osztályozót vagy regresszort tanítanak az egyedi preferenciák előrejelzésére. A mátrixfaktorizáció és az AutoRec pontonkénti célfüggvényekkel van optimalizálva. A páronkénti megközelítések minden felhasználóhoz elempárokat vizsgálnak, és az adott pár optimális sorrendjét próbálják közelíteni. A rangsorolási feladatokhoz általában a páronkénti megközelítések jobban illenek, mert a relatív sorrend előrejelzése közelebb áll a rangsorolás természetéhez. A lista-alapú megközelítések az elemek teljes listájának sorrendjét közelítik, például közvetlenül olyan rangsorolási mérőszámokat optimalizálnak, mint a Normalized Discounted Cumulative Gain ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). Ezek azonban összetettebbek és számításigényesebbek, mint a pontonkénti vagy páronkénti módszerek. Ebben a szakaszban két páronkénti célfüggvényt, a Bayes-féle személyre szabott rangsorolási veszteséget és a Hinge-veszteséget, valamint azok megvalósítását mutatjuk be.

## A Bayes-féle személyre szabott rangsorolási veszteség és megvalósítása

A Bayes-féle személyre szabott rangsorolás, röviden BPR :cite:`Rendle.Freudenthaler.Gantner.ea.2009`, egy páronkénti rangsorolási veszteség, amely a maximum a posteriori becslésből származik. Számos meglévő ajánlási modellben széles körben használják. A BPR tanítóadata pozitív és negatív párokból áll (a negatív példák a hiányzó értékekből származnak). Azt feltételezi, hogy a felhasználó a pozitív elemet előnyben részesíti az összes többi meg nem figyelt elemmel szemben.

Formálisan a tanítóadat $(u, i, j)$ alakú hármasokból áll, amelyek azt fejezik ki, hogy az $u$ felhasználó az $i$ elemet előnyben részesíti a $j$ elemmel szemben. A BPR Bayes-féle megfogalmazását, amely a hátsó valószínűség maximalizálására törekszik, az alábbiakban adjuk meg:

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

ahol $\Theta$ egy tetszőleges ajánlási modell paramétereit jelöli, a $>_u$ pedig az $u$ felhasználó számára kívánt személyre szabott teljes elemrangsorolást. A maximum a posteriori becslő segítségével levezethetjük a személyre szabott rangsorolási feladat általános optimalizálási kritériumát.

$$
\begin{aligned}
\textrm{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


ahol $D \stackrel{\textrm{def}}{=} \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ a tanítóhalmaz, $I^+_u$ az $u$ felhasználó által kedvelt elemek halmazát, $I$ az összes elemet, az $I \backslash I^+_u$ pedig a felhasználó által nem kedvelt többi elemet jelöli. A $\hat{y}_{ui}$ és $\hat{y}_{uj}$ az $u$ felhasználó $i$, illetve $j$ elemre vonatkozó előrejelzett pontszámai. A prior $p(\Theta)$ egy nulla várható értékű normális eloszlás, amelynek kovariancia-mátrixa $\Sigma_\Theta$. Itt $\Sigma_\Theta = \lambda_\Theta I$-t választjuk.

![A Bayes-féle személyre szabott rangsorolás szemléltetése](../img/rec-ranking.svg)
Az alaposztályként a `mxnet.gluon.loss.Loss`-t fogjuk használni, és a `forward` metódust felüldefiniálva állítjuk elő a Bayes-féle személyre szabott rangsorolási veszteséget. Először importáljuk a Loss osztályt és az np modult.

```{.python .input  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

A BPR-veszteség megvalósítása a következő.

```{.python .input  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Hinge-veszteség és megvalósítása

A rangsoroláshoz használt Hinge-veszteség eltér a gluon könyvtárban elérhető [hinge loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) alakjától, amelyet gyakran használnak például SVM-ekhez hasonló osztályozókban. Az ajánlórendszerekben rangsorolásra használt veszteség a következő alakú.

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

ahol $m$ a biztonsági ráhagyás mérete. Célja, hogy a negatív elemeket eltávolítsa a pozitív elemektől. A BPR-hez hasonlóan nem az abszolút kimeneteket, hanem a pozitív és negatív minták közötti releváns távolságot optimalizálja, ezért jól illeszkedik az ajánlórendszerekhez.

```{.python .input  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

Ez a két veszteség egymással felcserélhető személyre szabott rangsorolási feladatokban.

## Összefoglalás

- Az ajánlórendszerek személyre szabott rangsorolási feladataihoz háromféle rangsorolási veszteség használható: pontonkénti, páronkénti és lista-alapú módszerek.
- A két páronkénti veszteség, a Bayes-féle személyre szabott rangsorolási veszteség és a Hinge-veszteség, egymással felcserélhető.

## Gyakorlatok

- Léteznek a BPR-nek és a Hinge-veszteségnek változatai?
- Tudsz olyan ajánlási modelleket találni, amelyek BPR-t vagy Hinge-veszteséget használnak?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:
