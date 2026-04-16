# Terminológiai referencia

> **Kötelező érvényű az egész könyvben.**
> Ez a dokumentum az elfogadott terminológiai döntéseket rögzíti.
> Minden fejezet, függelék és megjegyzés szövegében kizárólag az alábbi magyar megfelelőket szabad használni.
> Kódblokkokban és API-nevekben az eredeti angol alak megtartandó.
> Lásd még: [WORKFLOW.md](WORKFLOW.md)

---

| Angol | Magyar | Megjegyzés |
|-------|--------|------------|
| activation | aktiváció | |
| activation function | aktivációs függvény | |
| activation layer | aktivációs réteg | |
| additive attention | additív figyelem | |
| affine transformation | affin transzformáció | |
| anchor box | horgonytéglalap | |
| API | API | megtartjuk angolul |
| attention head | figyelemfej | |
| attention mechanism | figyelemmechanizmus | nem: *figyelem mechanizmus*, nem: *figyelmi mechanizmus* |
| attention pooling | figyelempooling | prózában: *figyelemalapú pooling* |
| attention score | figyelempontszám | |
| attention weight | figyelemsúly | |
| autograd | autograd | megtartjuk angolul |
| automatic differentiation | automatikus differenciálás | |
| average pooling | átlagpooling | |
| backpropagation | visszaterjesztés | nem: *backpropagation* prózában |
| batch | batch | megtartjuk angolul |
| batch normalization | batchnormalizáció | nem: *batch normalizáció* (különírva) |
| batch size | batch méret | |
| bias (linear models) | eltolás | nem: *torzítás* ebben a kontextusban |
| bias (statistical) | torzítás | nem: *eltolás* ebben a kontextusban |
| bounding box | befoglaló téglalap | elfogadott rövidítés: *bbox* |
| broadcasting | kiterjesztés | zárójelben: *(Broadcasting)* az első előfordulásnál |
| causal mask | oksági maszk | |
| channel | csatorna | CNN-ek kontextusában |
| checkpoint | checkpoint | megtartjuk angolul |
| checkpointing | checkpointelés | |
| classifier | osztályozó | |
| compatibility function | illeszkedési függvény | nem: *kompatibilitási függvény* |
| computational graph | számítási gráf | |
| concatenation | összefűzés | |
| confidence interval | konfidenciaintervallum | |
| convolutional layer | konvolúciós réteg | |
| convolutional neural network (CNN) | konvolúciós neurális hálózat | |
| covariance | kovariancia | |
| cross-attention | keresztfigyelem | |
| cross-correlation | keresztkorreláció | |
| cross-entropy | keresztentrópia | |
| cumulative distribution function | eloszlásfüggvény | |
| dataset | adathalmaz | |
| decoder | dekódoló | |
| deep learning | mélytanulás / mélytanulási | |
| design matrix | tervmátrix | |
| deserialization | deszerializálás | |
| detach | leválaszt / leválasztás | |
| determinant | determináns | |
| dimension | dimenzió | egy tengely mérete |
| dimensionality | dimenziószám | nem: *dimenzió* ebben a kontextusban |
| dot product | skaláris szorzat | prózában; kódban: *dot product* megtartható |
| dot-product attention | skalárisszorzat-alapú figyelem | |
| downstream task | célfeladat | |
| dropout | dropout | megtartjuk angolul |
| eigendecomposition | sajátérték-felbontás | |
| eigenvalue | sajátérték | |
| eigenvector | sajátvektor | |
| embedding | beágyazás | |
| embedding dimension | beágyazási dimenzió | |
| empirical error | empirikus hiba | |
| encoder | kódoló | |
| entropy | entrópia | |
| epoch | epoch | megtartjuk angolul |
| estimator | becslő | |
| expectation | várható érték | |
| extension | bővítés | nem: *kiterjesztés* (azt a *broadcasting* foglalja) |
| feature | jellemző | |
| feature map | jellemzőtérkép | |
| fine-tuning | finomhangolás | |
| flatten | kiterít | |
| fork | fork | megtartjuk angolul; első előfordulás: *fork (elágaztatás)* |
| forward pass | előremenet | egyetlen előre irányú számítási lépés |
| forward propagation | előreterjesztés | nem: *előre terjedés*, nem: *előre-terjedés* |
| forward-mode differentiation | előre irányú differenciálás | |
| framework | keretrendszer | |
| fully connected layer | teljesen összekötött réteg | nem: *teljes összeköttetésű réteg* |
| generalization | általánosítás | |
| generative adversarial network (GAN) | generatív versengő hálózat | |
| gradient | gradiens | |
| gradient buffer | gradienspuffer | |
| gradient clipping | gradienslevágás | |
| gradient descent | gradienscsökkenés | nem: *gradiens ereszkedés*, nem: *gradiens descent* |
| gradient field | gradienstér | |
| gradient vector | gradiensvektor | |
| Hessian / Hessian matrix | Hesse-mátrix | |
| hidden layer | rejtett réteg | |
| hidden state | rejtett állapot | |
| hidden unit | rejtett egység | |
| holdout set | holdout halmaz | |
| hyperbolic tangent (tanh) | tanh (hiperbolikus tangens) függvény | |
| inference | inferencia | |
| inner product | belső szorzat | |
| input | bemenet | |
| intersection over union (IoU) | metszet per unió | elfogadott rövidítés: *IoU* |
| Jacobian / Jacobian matrix | Jacobi-mátrix | |
| kernel | kernel | megtartjuk angolul |
| key | kulcs | figyelemmechanizmus kontextusában |
| L2 norm | L2 norma | |
| language modeling | nyelvmodellezés | |
| layer normalization | rétegnormalizáció | |
| learning rate | tanulási ráta | nem: *tanulási sebesség* |
| learning rate decay | tanulási ráta csökkentése | |
| learning rate scheduler | tanulásiráta-ütemező | |
| likelihood | likelihood | megtartjuk angolul |
| line search | vonalkeresés | |
| log-likelihood | log-likelihood | megtartjuk angolul |
| logits | logitérték | |
| loss | veszteség | |
| loss function | veszteségfüggvény | |
| loss landscape | veszteségfelület | |
| machine learning | gépi tanulás | |
| masked self-attention | maszkolt önfigyelem | |
| matrix | mátrix | |
| maximum likelihood estimate | maximum likelihood becslés | nem: *maximális valószínűség*; rövidítés: MLE |
| mini-batch | mini-batch | megtartjuk angolul; nem: *minibatch* (egybe) |
| model | modell | |
| multi-head attention | többfejű figyelem | |
| multilayer perceptron (MLP) | többrétegű perceptron | rövidítés: MLP |
| negative log-likelihood | negatív log-likelihood | |
| neurons firing | kisül | nem: *tüzel* |
| non-informative prior | nem informatív prior | |
| non-maximum suppression (NMS) | nemmaximális elnyomás | elfogadott rövidítés: *NMS* |
| norm | norma | |
| normalization | normalizáció | |
| notebook | notebook | megtartjuk angolul |
| object detection | objektumdetektálás | nem: *objektumfelismerés* |
| optimizer | optimalizáló | |
| outer product | külső szorzat | |
| output | kimenet | |
| overfitting | túlillesztés | |
| padding | párnázás | konvolúciós hálózatok kontextusában |
| partial derivative | parciális derivált | |
| patch | patch | megtartjuk angolul (ViT-kontextusban) |
| pooling | pooling | megtartjuk angolul mélytanulási kontextusban |
| population | populáció | következetesen; alternatíva: *alapsokaság* |
| population error | populációs hiba | |
| positional encoding | pozicionális kódolás | |
| preconditioning | előkondicionálás | |
| pretraining | előtanítás | |
| prior | prior | megtartjuk angolul |
| probability density function | valószínűségi sűrűségfüggvény | |
| probability distribution | valószínűségi eloszlás | |
| probability mass function | valószínűségi tömegfüggvény | |
| pull request | pull request | megtartjuk angolul |
| query | lekérdezés | figyelemmechanizmus kontextusában |
| random variable | valószínűségi változó | |
| rank (matrix) | rang | |
| receptive field | receptív mező | |
| recurrent neural network (RNN) | rekurrens neurális hálózat | |
| rectified linear unit (ReLU) | ReLU | megtartjuk angolul |
| regularization | regularizáció | |
| repository | tároló | nem: *kódtár* Git/GitHub kontextusban |
| residual connection | maradékkapcsolat | |
| reverse-mode differentiation | visszafelé irányú differenciálás | |
| sample | minta | |
| scalar | skalár (fn.) / skaláris (mn.) | főnévként *skalár*, melléknévként *skaláris* |
| scaled dot-product attention | skálázott skalárisszorzat-alapú figyelem | |
| self-attention | önfigyelem | |
| sequence length | sorozathossz | |
| serialization | szerializálás | |
| sigmoid function | sigmoid függvény | |
| skip connection | áthidaló kapcsolat | |
| squashing function | értékkészletet szűkítő függvény | |
| standard deviation | szórás | |
| steepest descent direction | a legmeredekebb csökkenés iránya | negatív gradiens irányával azonos |
| stochastic gradient descent (SGD) | sztochasztikus gradienscsökkenés | nem: *sztochasztikus gradiens descent* |
| stride | lépésköz | konvolúciós hálózatok kontextusában |
| subword | részszó | |
| tensor | tenzor | |
| tensor shape | tenzor alakja | |
| test set | teszthalmaz | nem: *tesztkészlet* |
| time step | időlépés | szekvenciális modellek kontextusában |
| token | token | megtartjuk angolul |
| tokenization | tokenizálás | |
| trace (matrix) | nyom | nem: *trace* prózában |
| training | tanítás | |
| training set | tanítóhalmaz | nem: *tanítókészlet* |
| Transformer | Transformer | megtartjuk angolul; nem: *Transzformer* |
| underfitting | alulillesztés | |
| validation set | validációs halmaz | nem: *érvényesítési halmaz* |
| value | érték | figyelemmechanizmus kontextusában |
| variance | variancia | |
| vector | vektor | |
| vision Transformer | vision Transformer | |
| vocabulary | szókészlet | nem: *szókincskészlet* |
| weight decay | súlycsökkentés | |
