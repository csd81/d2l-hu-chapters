# Terminológiai referencia

> **Kötelező érvényű az egész könyvben.**
> Ez a dokumentum az elfogadott terminológiai döntéseket rögzíti.
> Minden fejezet, függelék és megjegyzés szövegében kizárólag az alábbi magyar megfelelőket szabad használni.
> Kódblokkokban és API-nevekben az eredeti angol alak megtartandó.

---

## 1. Lineáris algebra

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| determinant | determináns | |
| dimension | dimenzió | egy tengely mérete |
| dimensionality | dimenziószám | nem: *dimenzió* ebben a kontextusban |
| eigendecomposition | sajátérték-felbontás | |
| eigenvalue | sajátérték | |
| eigenvector | sajátvektor | |
| gradient field | gradienstér | |
| gradient vector | gradiensvektor | |
| Hessian / Hessian matrix | Hesse-mátrix | |
| inner product | belső szorzat | |
| Jacobian / Jacobian matrix | Jacobi-mátrix | |
| L2 norm | L2 norma | |
| matrix | mátrix | |
| norm | norma | |
| outer product | külső szorzat | |
| rank (matrix) | rang | |
| scalar | skalár (fn.) / skaláris (mn.) | főnévként *skalár*, melléknévként *skaláris* |
| tensor | tenzor | |
| tensor shape | tenzor alakja | |
| trace (matrix) | nyom | nem: *trace* prózában |
| vector | vektor | |

---

## 2. Valószínűségszámítás és statisztika

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| bias (statistical) | torzítás | nem: *eltolás* ebben a kontextusban |
| confidence interval | konfidenciaintervallum | |
| covariance | kovariancia | |
| cumulative distribution function | eloszlásfüggvény | |
| empirical error | empirikus hiba | |
| entropy | entrópia | |
| estimator | becslő | |
| expectation | várható érték | |
| likelihood | likelihood | megtartjuk angolul |
| log-likelihood | log-likelihood | megtartjuk angolul |
| maximum likelihood estimate | maximum likelihood becslés | nem: *maximális valószínűség*; rövidítés: MLE |
| negative log-likelihood | negatív log-likelihood | |
| non-informative prior | nem informatív prior | |
| partial derivative | parciális derivált | |
| population | populáció | következetesen; alternatíva: *alapsokaság* |
| population error | populációs hiba | |
| prior | prior | megtartjuk angolul |
| probability density function | valószínűségi sűrűségfüggvény | |
| probability distribution | valószínűségi eloszlás | |
| probability mass function | valószínűségi tömegfüggvény | |
| random variable | valószínűségi változó | |
| sample | minta | |
| standard deviation | szórás | |
| variance | variancia | |

---

## 3. Gépi tanulás — általános

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| affine transformation | affin transzformáció | |
| batch | batch | megtartjuk angolul |
| batch size | batch méret | |
| bias (linear models) | eltolás | nem: *torzítás* ebben a kontextusban |
| broadcasting | kiterjesztés | zárójelben: *(Broadcasting)* az első előfordulásnál |
| classifier | osztályozó | |
| concatenation | összefűzés | |
| dataset | adathalmaz | |
| deep learning | mélytanulás / mélytanulási | |
| design matrix | tervmátrix | |
| dropout | dropout | megtartjuk angolul |
| embedding | beágyazás | |
| epoch | epoch | megtartjuk angolul |
| extension | bővítés | nem: *kiterjesztés* (azt a *broadcasting* foglalja) |
| feature | jellemző | |
| fine-tuning | finomhangolás | |
| flatten | kiterít | |
| generalization | általánosítás | |
| holdout set | holdout halmaz | |
| inference | inferencia | |
| machine learning | gépi tanulás | |
| mini-batch | mini-batch | megtartjuk angolul; nem: *minibatch* (egybe) |
| model | modell | |
| normalization | normalizáció | |
| overfitting | túlillesztés | |
| pretraining | előtanítás | |
| regularization | regularizáció | |
| test set | teszthalmaz | nem: *tesztkészlet* |
| training | tanítás | |
| training set | tanítóhalmaz | nem: *tanítókészlet* |
| underfitting | alulillesztés | |
| validation set | validációs halmaz | nem: *érvényesítési halmaz* |
| weight decay | súlycsökkentés | |

---

## 4. Neurális hálózatok

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| activation | aktiváció | |
| activation function | aktivációs függvény | |
| activation layer | aktivációs réteg | |
| backpropagation | visszaterjesztés | nem: *backpropagation* prózában |
| batch normalization | batchnormalizáció | nem: *batch normalizáció* (különírva) |
| channel | csatorna | CNN-ek kontextusában |
| convolutional layer | konvolúciós réteg | |
| convolutional neural network (CNN) | konvolúciós neurális hálózat | |
| cross-correlation | keresztkorreláció | |
| decoder | dekódoló | |
| encoder | kódoló | |
| feature map | jellemzőtérkép | |
| forward pass | előremenet | egyetlen előre irányú számítási lépés |
| forward propagation | előreterjesztés | nem: *előre terjedés*, nem: *előre-terjedés* |
| forward-mode differentiation | előre irányú differenciálás | |
| fully connected layer | teljesen összekötött réteg | nem: *teljes összeköttetésű réteg* |
| hidden layer | rejtett réteg | |
| hidden state | rejtett állapot | |
| hidden unit | rejtett egység | |
| hyperbolic tangent (tanh) | tanh (hiperbolikus tangens) függvény | |
| input | bemenet | |
| kernel | kernel | megtartjuk angolul |
| layer normalization | rétegnormalizáció | |
| logits | logitérték | |
| multilayer perceptron (MLP) | többrétegű perceptron | rövidítés: MLP |
| neurons firing | kisül | nem: *tüzel* |
| output | kimenet | |
| padding | párnázás | konvolúciós hálózatok kontextusában |
| pooling | pooling | megtartjuk angolul mélytanulási kontextusban |
| receptive field | receptív mező | |
| recurrent neural network (RNN) | rekurrens neurális hálózat | |
| rectified linear unit (ReLU) | ReLU | megtartjuk angolul |
| residual connection | maradékkapcsolat | |
| sigmoid function | sigmoid függvény | |
| skip connection | áthidaló kapcsolat | |
| squashing function | értékkészletet szűkítő függvény | |
| stride | lépésköz | konvolúciós hálózatok kontextusában |
| time step | időlépés | szekvenciális modellek kontextusában |

---

## 5. Optimalizálás és automatikus differenciálás

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| automatic differentiation | automatikus differenciálás | |
| autograd | autograd | megtartjuk angolul |
| computational graph | számítási gráf | |
| detach | leválaszt / leválasztás | |
| gradient | gradiens | |
| gradient buffer | gradienspuffer | |
| gradient clipping | gradienslevágás | |
| gradient descent | gradienscsökkenés | nem: *gradiens ereszkedés*, nem: *gradiens descent* |
| learning rate | tanulási ráta | nem: *tanulási sebesség* |
| learning rate decay | tanulási ráta csökkentése | |
| learning rate scheduler | tanulásiráta-ütemező | |
| line search | vonalkeresés | |
| loss | veszteség | |
| loss function | veszteségfüggvény | |
| loss landscape | veszteségfelület | |
| optimizer | optimalizáló | |
| preconditioning | előkondicionálás | |
| reverse-mode differentiation | visszafelé irányú differenciálás | |
| steepest descent direction | a legmeredekebb csökkenés iránya | negatív gradiens irányával azonos |
| stochastic gradient descent (SGD) | sztochasztikus gradienscsökkenés | nem: *sztochasztikus gradiens descent* |

---

## 6. Figyelemmechanizmus és Transformer

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| additive attention | additív figyelem | |
| attention head | figyelemfej | |
| attention mechanism | figyelemmechanizmus | nem: *figyelem mechanizmus*, nem: *figyelmi mechanizmus* |
| attention pooling | figyelempooling | prózában: *figyelemalapú pooling* |
| attention score | figyelempontszám | |
| attention weight | figyelemsúly | |
| average pooling | átlagpooling | |
| causal mask | oksági maszk | |
| compatibility function | illeszkedési függvény | nem: *kompatibilitási függvény* |
| cross-attention | keresztfigyelem | |
| dot product | skaláris szorzat | prózában; kódban: *dot product* megtartható |
| dot-product attention | skalárisszorzat-alapú figyelem | |
| key | kulcs | figyelemmechanizmus kontextusában |
| masked self-attention | maszkolt önfigyelem | |
| multi-head attention | többfejű figyelem | |
| positional encoding | pozicionális kódolás | |
| query | lekérdezés | figyelemmechanizmus kontextusában |
| scaled dot-product attention | skálázott skalárisszorzat-alapú figyelem | |
| self-attention | önfigyelem | |
| sequence length | sorozathossz | |
| Transformer | Transformer | megtartjuk angolul; nem: *Transzformer* |
| value | érték | figyelemmechanizmus kontextusában |
| vision Transformer | vision Transformer | |

---

## 7. Természetes nyelvfeldolgozás (NLP)

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| downstream task | célfeladat | |
| embedding dimension | beágyazási dimenzió | |
| language modeling | nyelvmodellezés | |
| subword | részszó | |
| token | token | megtartjuk angolul |
| tokenization | tokenizálás | |
| vocabulary | szókészlet | nem: *szókincskészlet* |

---

## 8. Számítógépes látás

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| anchor box | horgonytéglalap | |
| bounding box | befoglaló téglalap | elfogadott rövidítés: *bbox* |
| intersection over union (IoU) | metszet per unió | elfogadott rövidítés: *IoU* |
| non-maximum suppression (NMS) | nemmaximális elnyomás | elfogadott rövidítés: *NMS* |
| object detection | objektumdetektálás | nem: *objektumfelismerés* |

---

## 9. Szoftver és mérnöki szöveg

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| API | API | megtartjuk angolul |
| checkpoint | checkpoint | megtartjuk angolul |
| checkpointing | checkpointelés | |
| deserialization | deszerializálás | |
| fork | fork | megtartjuk angolul; első előfordulás: *fork (elágaztatás)* |
| framework | keretrendszer | |
| generative adversarial network (GAN) | generatív versengő hálózat | |
| notebook | notebook | megtartjuk angolul |
| patch | patch | megtartjuk angolul (ViT-kontextusban) |
| pull request | pull request | megtartjuk angolul |
| repository | tároló | nem: *kódtár* Git/GitHub kontextusban |
| serialization | szerializálás | |
