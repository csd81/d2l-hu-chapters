# Terminológiai referencia

> **Kötelező érvényű az egész könyvben.**
> Ez a dokumentum az elfogadott terminológiai döntéseket rögzíti.
> Minden fejezet, függelék és megjegyzés szövegében kizárólag az alábbi magyar megfelelőket szabad használni.
> Kódblokkokban és API-nevekben az eredeti angol alak megtartandó.

---

## Elfogadott terminológia (angol → magyar)

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| additive attention | additív figyelem | |
| anchor box | horgonytéglalap | |
| attention mechanism | figyelemmechanizmus | nem: *figyelem mechanizmus*, nem: *figyelmi mechanizmus* |
| attention pooling | figyelempooling | prózában: *figyelemalapú pooling* |
| attention score | figyelempontszám | |
| attention weight | figyelemsúly | |
| autograd | autograd | megtartjuk angolul |
| automatic differentiation | automatikus differenciálás | |
| average pooling | átlagpooling | |
| backpropagation | visszaterjesztés | nem: *backpropagation* prózában |
| batch | batch | megtartjuk angolul |
| batch normalization | batchnormalizáció | |
| bias (linear models) | eltolás | nem: *torzítás* ebben a kontextusban |
| bias (statistical) | torzítás | nem: *eltolás* ebben a kontextusban |
| bounding box | befoglaló téglalap | elfogadott rövidítés: *bbox* |
| broadcasting | kiterjesztés | zárójelben: *(Broadcasting)* az első előfordulásnál |
| checkpoint | checkpoint | megtartjuk angolul |
| compatibility function | illeszkedési függvény | nem: *kompatibilitási függvény* |
| computational graph | számítási gráf | |
| confidence interval | konfidenciaintervallum | |
| cross-attention | keresztfigyelem | |
| cross-correlation | keresztkorreláció | |
| cross-entropy | keresztentrópia | |
| cumulative distribution function | eloszlásfüggvény | |
| dataset | adathalmaz | |
| decoder | dekódoló | |
| deep learning | mélytanulás / mélytanulási | |
| detach | leválaszt / leválasztás | |
| dimensionality | dimenzió / dimenziószám | |
| dot product | skaláris szorzat | prózában; kódban: *dot product* megtartható |
| dot-product attention | skalárisszorzat-alapú figyelem | |
| downstream task | célfeladat | |
| dropout | dropout | megtartjuk angolul |
| eigendecomposition | sajátérték-felbontás | |
| eigenvalue | sajátérték | |
| eigenvector | sajátvektor | |
| embedding | beágyazás | |
| encoder | kódoló | |
| entropy | entrópia | |
| epoch | epoch | megtartjuk angolul |
| estimator | becslő | |
| feature | jellemző | |
| feature map | jellemzőtérkép | |
| fine-tuning | finomhangolás | |
| fork | fork | megtartjuk angolul; első előfordulás: *fork (elágaztatás)* |
| forward pass | előremenet | egyetlen előre irányú számítási lépés |
| forward propagation | előreterjesztés | nem: *előre terjedés*, nem: *előre-terjedés* |
| forward-mode differentiation | előre irányú differenciálás | |
| framework | keretrendszer | |
| fully connected layer | teljesen összekötött réteg | nem: *teljes összeköttetésű réteg* |
| generative adversarial network (GAN) | generatív versengő hálózat | |
| gradient | gradiens | |
| gradient buffer | gradienspuffer | |
| gradient descent | gradienscsökkenés | nem: *gradiens ereszkedés*, nem: *gradiens descent* |
| Hessian / Hessian matrix | Hesse-mátrix | |
| hidden state | rejtett állapot | |
| input | bemenet | |
| intersection over union (IoU) | metszet per unió | elfogadott rövidítés: *IoU* |
| Jacobian / Jacobian matrix | Jacobi-mátrix | |
| kernel | kernel | megtartjuk angolul |
| key | kulcs | figyelemmechanizmus kontextusában |
| layer normalization | rétegnormalizáció | |
| learning rate | tanulási ráta | nem: *tanulási sebesség* |
| learning rate scheduler | tanulásiráta-ütemező | |
| likelihood | likelihood | megtartjuk angolul |
| line search | vonalkeresés | |
| log-likelihood | log-likelihood | megtartjuk angolul |
| loss | veszteség | |
| loss function | veszteségfüggvény | |
| machine learning | gépi tanulás | |
| masked self-attention | maszkolt önfigyelem | |
| matrix | mátrix | |
| maximum likelihood estimate | maximum likelihood becslés | nem: *maximális valószínűség* |
| mini-batch | mini-batch | megtartjuk angolul |
| multi-head attention | többfejű figyelem | |
| negative log-likelihood | negatív log-likelihood | |
| neurons firing | kisül | nem: *tüzel* |
| non-informative prior | nem informatív prior | |
| non-maximum suppression (NMS) | nemmaximális elnyomás | elfogadott rövidítés: *NMS* |
| notebook | notebook | megtartjuk angolul |
| object detection | objektumdetektálás | nem: *objektumfelismerés* |
| output | kimenet | |
| partial derivative | parciális derivált | |
| padding | párnázás | konvolúciós hálózatok kontextusában |
| patch | patch | megtartjuk angolul (ViT-kontextusban) |
| pooling | pooling | megtartjuk angolul mélytanulási kontextusban |
| population | populáció | következetesen; alternatíva: *alapsokaság* |
| positional encoding | pozicionális kódolás | |
| preconditioning | előkondicionálás | |
| pretraining | előtanítás | |
| prior | prior | megtartjuk angolul |
| probability density function | valószínűségi sűrűségfüggvény | |
| probability mass function | valószínűségi tömegfüggvény | |
| pull request | pull request | megtartjuk angolul |
| query | lekérdezés | figyelemmechanizmus kontextusában |
| random variable | valószínűségi változó | |
| receptive field | receptív mező | |
| regularization | regularizáció | |
| repository | tároló | nem: *kódtár* Git/GitHub kontextusban |
| reverse-mode differentiation | visszafelé irányú differenciálás | |
| sample | minta | |
| scalar | skalár (fn.) / skaláris (mn.) | főnévként *skalár*, melléknévként *skaláris* |
| scaled dot-product attention | skálázott skalárisszorzat-alapú figyelem | |
| self-attention | önfigyelem | |
| squashing function | értékkészletet szűkítő függvény | |
| steepest descent direction | a legmeredekebb csökkenés iránya | negatív gradiens irányával azonos |
| stochastic gradient descent (SGD) | sztochasztikus gradienscsökkenés | nem: *sztochasztikus gradiens descent* |
| stride | lépésköz | konvolúciós hálózatok kontextusában |
| tensor | tenzor | |
| tensor shape | tenzor alakja | |
| test set | teszthalmaz | nem: *tesztkészlet* |
| token | token | megtartjuk angolul |
| training set | tanítóhalmaz | nem: *tanítókészlet* |
| Transformer | Transformer | megtartjuk angolul; nem: *Transzformer* |
| validation set | validációs halmaz | nem: *érvényesítési halmaz* |
| value | érték | figyelemmechanizmus kontextusában |
| vector | vektor | |
| vision Transformer | vision Transformer | |
| vocabulary | szókészlet | |
| weight decay | súlycsökkentés | |
