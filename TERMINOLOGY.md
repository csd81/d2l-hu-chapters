# Terminológiai referencia

## Cél

**Globális konzisztencia.** Minden fejezet, függelék és megjegyzés szövege egyazon magyar terminológiát használja. Az olvasó soha nem találkozhat azzal, hogy ugyanazt a fogalmat az egyik fejezetben `gradienscsökkenés`-nek, a másikban `gradiens ereszkedés`-nek hívják.

Az egyetlen hiteles forrás: ez a fájl.

---

## Szabályok

1. **Ha egy fogalom szerepel a táblázatban**, azt a táblázatban szereplő magyar alakban kell használni — kivételek nincsenek.
2. **Ha egy fogalom nincs a táblázatban**, először fel kell venni, és csak utána szabad a fejezetbe írni.
3. **Kódblokkokban** (```` ``` ```` és `` ` `` között) az eredeti angol API-nevek és változónevek maradnak.
4. **Kódkommentekben** a szöveg a táblázat alapján magyarul írandó.
5. **Angol megjelenés prózában** csak akkor engedett, ha a `Megjegyzés` oszlopban explicit `megtartjuk angolul` jelölés szerepel.

---

## Új terminus felvétele

1. Ellenőrizd, hogy a fogalom nincs-e már a táblázatban.
2. Adj hozzá egy sort alfabetikus sorrendben.
3. Ha a döntés vitatható (két elfogadott alak létezik), add meg a `nem:` megszorítást is.
4. Commit ezt a fájlt külön, a fejezet előtt — így a változás nyomon követhető.

---

## Fejezet ellenőrzése (manuális)

Mielőtt egy fejezet módosítása bekerül a repóba:

```bash
grep -rn "gradiens ereszkedés" chapter_*/
grep -rn "tanulási sebesség" chapter_*/
grep -rn "figyelem mechanizmus" chapter_*/
grep -rn "minibatch[^-]" chapter_*/
grep -rn "torzítás" chapter_*/   # csak lineáris modell kontextusban ellenőrizd
```

Ha találat van, javítsd a táblázat alapján, mielőtt pusholsz.

---

## Fejezetek ellenőrzési sorrendje

1. `chapter_appendix-mathematics-for-deep-learning/`
2. `chapter_optimization/`
3. `chapter_linear-regression/`
4. `chapter_linear-classification/`
5. `chapter_multilayer-perceptrons/`
6. `chapter_convolutional-neural-networks/`
7. `chapter_convolutional-modern/`
8. `chapter_recurrent-neural-networks/`
9. `chapter_recurrent-modern/`
10. `chapter_attention-mechanisms-and-transformers/`
11. `chapter_natural-language-processing-pretraining/`
12. `chapter_natural-language-processing-applications/`
13. `chapter_computer-vision/`
14. `chapter_builders-guide/`
15. `chapter_computational-performance/`
16. `chapter_recommender-systems/`
17. `chapter_reinforcement-learning/`
18. `chapter_gaussian-processes/`
19. `chapter_appendix-tools-for-deep-learning/`

---

## Elvégzett pass-ok (napló)

| Dátum      | Fejezetek            | Érintett fájlok | Fő változások                                                                                             |
| ---------- | -------------------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| 2026-04-16 | Teljes repó          | 71              | Alapszótár felépítése; gradienscsökkenés, figyelemmechanizmus, tanítóhalmaz stb. egységesítése            |
| 2026-04-16 | Teljes repó          | 29              | tanulási ráta, előreterjesztés, objektumdetektálás, autograd.md javítások                                 |
| 2026-04-16 | conv, rnn, attention | 5               | padding/stride/CNN/RNN felvétele; kötegméret, dot product figyelem, title-case javítások                  |
| 2026-04-16 | Teljes repó          | 71              | mini-batch egységesítés (49 fájl), batchnormalizáció, torzítás→eltolás (25 fájl), szókincskészlet javítás |
| 2026-04-16 | Teljes repó          | 32              | Linter hozzáadva (scripts/check_terminology.py + .pre-commit-config.yaml); 97 maradvány: Transzformer→Transformer (17), előre irányú terjesztés→előreterjesztés (14), tanítási/tesztelési halmaz→kanonikus alak (30+), backpropagation→visszaterjesztés (7) |
| 2026-04-16 | Teljes repó          | 19              | Denylist bővítve (enkóder→kódoló, dekóder→dekódoló, batch normalizálás, visszafelé irányú terjesztés, előrepasszolás, gradiens módszer, forward propagation); 84 maradvány javítva; szerkesztési hiba és elírás javítva (bounding-box.md, rcnn.md) |
| 2026-04-16 | Teljes repó          | 25              | Denylist bővítve (rétegnormalizálás, tanítási/tesztadathalmaz, előre irányú számítás/folyamat, forward propagáció); 46 maradvány javítva; 5 suffix-artifact manuálisan javítva (transformer.md, large-pretraining-transformers.md, hardware.md, batch-norm.md) |

---

## Szójegyzék

| Angol                                | Magyar                                   | Megjegyzés                                                 |
| ------------------------------------ | ---------------------------------------- | ---------------------------------------------------------- |
| activation                           | aktiváció                                |                                                            |
| activation function                  | aktivációs függvény                      |                                                            |
| activation layer                     | aktivációs réteg                         |                                                            |
| additive attention                   | additív figyelem                         |                                                            |
| affine transformation                | affin transzformáció                     |                                                            |
| anchor box                           | horgonytéglalap                          |                                                            |
| API                                  | API                                      | megtartjuk angolul                                         |
| attention head                       | figyelemfej                              |                                                            |
| attention mechanism                  | figyelemmechanizmus                      | nem: *figyelem mechanizmus*, nem: *figyelmi mechanizmus*   |
| attention pooling                    | figyelempooling                          | prózában: *figyelemalapú pooling*                          |
| attention score                      | figyelempontszám                         |                                                            |
| attention weight                     | figyelemsúly                             |                                                            |
| autograd                             | autograd                                 | megtartjuk angolul                                         |
| automatic differentiation            | automatikus differenciálás               |                                                            |
| average pooling                      | átlagpooling                             |                                                            |
| backpropagation                      | visszaterjesztés                         | nem: *backpropagation* prózában                            |
| batch                                | batch                                    | megtartjuk angolul                                         |
| batch normalization                  | batchnormalizáció                        | nem: *batch normalizáció* (különírva)                      |
| batch size                           | batch méret                              |                                                            |
| bias (linear models)                 | eltolás                                  | nem: *torzítás* ebben a kontextusban                       |
| bias (statistical)                   | torzítás                                 | nem: *eltolás* ebben a kontextusban                        |
| bounding box                         | befoglaló téglalap                       | elfogadott rövidítés: *bbox*                               |
| broadcasting                         | kiterjesztés                             | zárójelben: *(Broadcasting)* az első előfordulásnál        |
| causal mask                          | oksági maszk                             |                                                            |
| channel                              | csatorna                                 | CNN-ek kontextusában                                       |
| checkpoint                           | checkpoint                               | megtartjuk angolul                                         |
| checkpointing                        | checkpointelés                           |                                                            |
| classifier                           | osztályozó                               |                                                            |
| compatibility function               | illeszkedési függvény                    | nem: *kompatibilitási függvény*                            |
| computational graph                  | számítási gráf                           |                                                            |
| concatenation                        | összefűzés                               |                                                            |
| confidence interval                  | konfidenciaintervallum                   |                                                            |
| convolutional layer                  | konvolúciós réteg                        |                                                            |
| convolutional neural network (CNN)   | konvolúciós neurális hálózat             |                                                            |
| covariance                           | kovariancia                              |                                                            |
| cross-attention                      | keresztfigyelem                          |                                                            |
| cross-correlation                    | keresztkorreláció                        |                                                            |
| cross-entropy                        | keresztentrópia                          |                                                            |
| cumulative distribution function     | eloszlásfüggvény                         |                                                            |
| dataset                              | adathalmaz                               |                                                            |
| decoder                              | dekódoló                                 |                                                            |
| deep learning                        | mélytanulás / mélytanulási               |                                                            |
| design matrix                        | tervmátrix                               |                                                            |
| deserialization                      | deszerializálás                          |                                                            |
| detach                               | leválaszt / leválasztás                  |                                                            |
| determinant                          | determináns                              |                                                            |
| dimension                            | dimenzió                                 | egy tengely mérete                                         |
| dimensionality                       | dimenziószám                             | nem: *dimenzió* ebben a kontextusban                       |
| dot product                          | skaláris szorzat                         | prózában; kódban: *dot product* megtartható                |
| dot-product attention                | skalárisszorzat-alapú figyelem           |                                                            |
| downstream task                      | célfeladat                               |                                                            |
| dropout                              | dropout                                  | megtartjuk angolul                                         |
| eigendecomposition                   | sajátérték-felbontás                     |                                                            |
| eigenvalue                           | sajátérték                               |                                                            |
| eigenvector                          | sajátvektor                              |                                                            |
| embedding                            | beágyazás                                |                                                            |
| embedding dimension                  | beágyazási dimenzió                      |                                                            |
| empirical error                      | empirikus hiba                           |                                                            |
| encoder                              | kódoló                                   |                                                            |
| entropy                              | entrópia                                 |                                                            |
| epoch                                | epoch                                    | megtartjuk angolul                                         |
| estimator                            | becslő                                   |                                                            |
| expectation                          | várható érték                            |                                                            |
| extension                            | bővítés                                  | nem: *kiterjesztés* (azt a *broadcasting* foglalja)        |
| feature                              | jellemző                                 |                                                            |
| feature map                          | jellemzőtérkép                           |                                                            |
| fine-tuning                          | finomhangolás                            |                                                            |
| flatten                              | kiterít                                  |                                                            |
| fork                                 | fork                                     | megtartjuk angolul; első előfordulás: *fork (elágaztatás)* |
| forward pass                         | előremenet                               | egyetlen előre irányú számítási lépés                      |
| forward propagation                  | előreterjesztés                          | nem: *előre terjedés*, nem: *előre-terjedés*               |
| forward-mode differentiation         | előre irányú differenciálás              |                                                            |
| framework                            | keretrendszer                            |                                                            |
| fully connected layer                | teljesen összekötött réteg               | nem: *teljes összeköttetésű réteg*                         |
| generalization                       | általánosítás                            |                                                            |
| generative adversarial network (GAN) | generatív versengő hálózat               |                                                            |
| gradient                             | gradiens                                 |                                                            |
| gradient buffer                      | gradienspuffer                           |                                                            |
| gradient clipping                    | gradienslevágás                          |                                                            |
| gradient descent                     | gradienscsökkenés                        | nem: *gradiens ereszkedés*, nem: *gradiens descent*        |
| gradient field                       | gradienstér                              |                                                            |
| gradient vector                      | gradiensvektor                           |                                                            |
| Hessian / Hessian matrix             | Hesse-mátrix                             |                                                            |
| hidden layer                         | rejtett réteg                            |                                                            |
| hidden state                         | rejtett állapot                          |                                                            |
| hidden unit                          | rejtett egység                           |                                                            |
| holdout set                          | holdout halmaz                           |                                                            |
| hyperbolic tangent (tanh)            | tanh (hiperbolikus tangens) függvény     |                                                            |
| inference                            | inferencia                               |                                                            |
| inner product                        | belső szorzat                            |                                                            |
| input                                | bemenet                                  |                                                            |
| intersection over union (IoU)        | metszet per unió                         | elfogadott rövidítés: *IoU*                                |
| Jacobian / Jacobian matrix           | Jacobi-mátrix                            |                                                            |
| kernel                               | kernel                                   | megtartjuk angolul                                         |
| key                                  | kulcs                                    | figyelemmechanizmus kontextusában                          |
| L2 norm                              | L2 norma                                 |                                                            |
| language modeling                    | nyelvmodellezés                          |                                                            |
| layer normalization                  | rétegnormalizáció                        |                                                            |
| learning rate                        | tanulási ráta                            | nem: *tanulási sebesség*                                   |
| learning rate decay                  | tanulási ráta csökkentése                |                                                            |
| learning rate scheduler              | tanulásiráta-ütemező                     |                                                            |
| likelihood                           | likelihood                               | megtartjuk angolul                                         |
| line search                          | vonalkeresés                             |                                                            |
| log-likelihood                       | log-likelihood                           | megtartjuk angolul                                         |
| logits                               | logitérték                               |                                                            |
| loss                                 | veszteség                                |                                                            |
| loss function                        | veszteségfüggvény                        |                                                            |
| loss landscape                       | veszteségfelület                         |                                                            |
| machine learning                     | gépi tanulás                             |                                                            |
| masked self-attention                | maszkolt önfigyelem                      |                                                            |
| matrix                               | mátrix                                   |                                                            |
| maximum likelihood estimate          | maximum likelihood becslés               | nem: *maximális valószínűség*; rövidítés: MLE              |
| mini-batch                           | mini-batch                               | megtartjuk angolul; nem: *minibatch* (egybe)               |
| model                                | modell                                   |                                                            |
| multi-head attention                 | többfejű figyelem                        |                                                            |
| multilayer perceptron (MLP)          | többrétegű perceptron                    | rövidítés: MLP                                             |
| negative log-likelihood              | negatív log-likelihood                   |                                                            |
| neurons firing                       | kisül                                    | nem: *tüzel*                                               |
| non-informative prior                | nem informatív prior                     |                                                            |
| non-maximum suppression (NMS)        | nemmaximális elnyomás                    | elfogadott rövidítés: *NMS*                                |
| norm                                 | norma                                    |                                                            |
| normalization                        | normalizáció                             |                                                            |
| notebook                             | notebook                                 | megtartjuk angolul                                         |
| object detection                     | objektumdetektálás                       | nem: *objektumfelismerés*                                  |
| optimizer                            | optimalizáló                             |                                                            |
| outer product                        | külső szorzat                            |                                                            |
| output                               | kimenet                                  |                                                            |
| overfitting                          | túlillesztés                             |                                                            |
| padding                              | párnázás                                 | konvolúciós hálózatok kontextusában                        |
| partial derivative                   | parciális derivált                       |                                                            |
| patch                                | patch                                    | megtartjuk angolul (ViT-kontextusban)                      |
| pooling                              | pooling                                  | megtartjuk angolul mélytanulási kontextusban               |
| population                           | populáció                                | következetesen; alternatíva: *alapsokaság*                 |
| population error                     | populációs hiba                          |                                                            |
| positional encoding                  | pozicionális kódolás                     |                                                            |
| preconditioning                      | előkondicionálás                         |                                                            |
| pretraining                          | előtanítás                               |                                                            |
| prior                                | prior                                    | megtartjuk angolul                                         |
| probability density function         | valószínűségi sűrűségfüggvény            |                                                            |
| probability distribution             | valószínűségi eloszlás                   |                                                            |
| probability mass function            | valószínűségi tömegfüggvény              |                                                            |
| pull request                         | pull request                             | megtartjuk angolul                                         |
| query                                | lekérdezés                               | figyelemmechanizmus kontextusában                          |
| random variable                      | valószínűségi változó                    |                                                            |
| rank (matrix)                        | rang                                     |                                                            |
| receptive field                      | receptív mező                            |                                                            |
| recurrent neural network (RNN)       | rekurrens neurális hálózat               |                                                            |
| rectified linear unit (ReLU)         | ReLU                                     | megtartjuk angolul                                         |
| regularization                       | regularizáció                            |                                                            |
| repository                           | tároló                                   | nem: *kódtár* Git/GitHub kontextusban                      |
| residual connection                  | maradékkapcsolat                         |                                                            |
| reverse-mode differentiation         | visszafelé irányú differenciálás         |                                                            |
| sample                               | minta                                    |                                                            |
| scalar                               | skalár (fn.) / skaláris (mn.)            | főnévként *skalár*, melléknévként *skaláris*               |
| scaled dot-product attention         | skálázott skalárisszorzat-alapú figyelem |                                                            |
| self-attention                       | önfigyelem                               |                                                            |
| sequence length                      | sorozathossz                             |                                                            |
| serialization                        | szerializálás                            |                                                            |
| sigmoid function                     | sigmoid függvény                         |                                                            |
| skip connection                      | áthidaló kapcsolat                       |                                                            |
| squashing function                   | értékkészletet szűkítő függvény          |                                                            |
| standard deviation                   | szórás                                   |                                                            |
| steepest descent direction           | a legmeredekebb csökkenés iránya         | negatív gradiens irányával azonos                          |
| stochastic gradient descent (SGD)    | sztochasztikus gradienscsökkenés         | nem: *sztochasztikus gradiens descent*                     |
| stride                               | lépésköz                                 | konvolúciós hálózatok kontextusában                        |
| subword                              | részszó                                  |                                                            |
| tensor                               | tenzor                                   |                                                            |
| tensor shape                         | tenzor alakja                            |                                                            |
| test set                             | teszthalmaz                              | nem: *tesztkészlet*                                        |
| time step                            | időlépés                                 | szekvenciális modellek kontextusában                       |
| token                                | token                                    | megtartjuk angolul                                         |
| tokenization                         | tokenizálás                              |                                                            |
| trace (matrix)                       | nyom                                     | nem: *trace* prózában                                      |
| training                             | tanítás                                  |                                                            |
| training set                         | tanítóhalmaz                             | nem: *tanítókészlet*                                       |
| Transformer                          | Transformer                              | megtartjuk angolul; nem: *Transzformer*                    |
| underfitting                         | alulillesztés                            |                                                            |
| validation set                       | validációs halmaz                        | nem: *érvényesítési halmaz*                                |
| value                                | érték                                    | figyelemmechanizmus kontextusában                          |
| variance                             | variancia                                |                                                            |
| vector                               | vektor                                   |                                                            |
| vision Transformer                   | vision Transformer                       |                                                            |
| vocabulary                           | szókészlet                               | nem: *szókincskészlet*                                     |
| weight decay                         | súlycsökkentés                           |                                                            |
