# d2l-hu-chapters

## Terminológiai döntések és glosszárium

### Terminológiai döntések

- **Broadcasting**: magyar szövegben egységesen *kiterjesztés* (zárójelben: Broadcasting) formában használjuk.
- **Bias (b paraméter)**: lineáris modellekben egységesen *eltolás*.
- **Bias (statisztikai)**: a torzítás–variancia átváltás kontextusában *torzítás*.
- **Gradient Descent**: egységesen *gradienscsökkenés* (nem *gradiens ereszkedés*).
- **Stochastic Gradient Descent (SGD)**: egységesen *sztochasztikus gradienscsökkenés*.
- **Deep learning**: szövegben egységesen *mélytanulás* / *mélytanulási*.
- **Machine learning**: szövegben egységesen *gépi tanulás*.
- **Attention mechanism**: egységesen *figyelemmechanizmus* (nem *figyelem mechanizmus*, nem *figyelmi mechanizmus*).
- **Self-attention**: egységesen *önfigyelem*.
- **Multi-head attention**: egységesen *többfejű figyelem*.
- **Positional encoding**: egységesen *pozicionális kódolás*.
- **Dot product**: prózában egységesen *skaláris szorzat*.
- **Loss function / loss**: *veszteségfüggvény* / *veszteség*.
- **Training set**: egységesen *tanítóhalmaz* (nem *tanítókészlet*).
- **Validation set**: egységesen *validációs halmaz* (nem *érvényesítési halmaz*).
- **Test set**: egységesen *teszthalmaz* (nem *tesztkészlet*).
- **Dataset**: egységesen *adathalmaz*.
- **Embedding**: egységesen *beágyazás*.
- **Fine-tuning**: egységesen *finomhangolás*.
- **Pretraining**: egységesen *előtanítás*.
- **Feature map**: egységesen *jellemzőtérkép*.
- **Weight decay**: egységesen *súlycsökkentés*.
- **Regularization**: egységesen *regularizáció*.
- **Batch normalization**: egységesen *batchnormalizáció*.
- **Layer normalization**: egységesen *rétegnormalizáció*.
- **Generative Adversarial Network (GAN)**: egységesen *generatív versengő hálózat*.
- **Squashing function**: magyarul *értékkészletet szűkítő függvény*.
- **Maximum likelihood**: egységesen *maximum likelihood becslés* (nem *maximális valószínűség*).
- **Eigenvalue / Eigenvector**: egységesen *sajátérték / sajátvektor*.
- **Eigendecomposition**: egységesen *sajátérték-felbontás*.
- **Random variable**: egységesen *valószínűségi változó*.
- **Confidence interval**: egységesen *konfidenciaintervallum*.
- **Encoder / Decoder**: egységesen *kódoló / dekódoló*.
- **Query / Key / Value**: egységesen *lekérdezés / kulcs / érték*.
- **Transformer**: megtartjuk *Transformer* alakban (nem *Transzformer*).
- **Input/Output**: szövegben egységesen *bemenet/kimenet*.
- **Scalar**: főnévként *skalár* (többes számban: *skalárok*), a *skaláris* alakot melléknévként használjuk.
- **Dimensionality**: magyar matematikai szövegben *dimenzió* vagy *dimenziószám*.
- **Neurons firing**: idegtudományi kontextusban *kisül* (nem *tüzel*).
- **Megtartjuk angolul** (kódolói közösségben elfogadott terminus): *dropout*, *batch*, *mini-batch*, *epoch*, *patch* (ViT-kontextusban), *notebook*, *kernel*, *fork*, *pull request*.

### Glosszárium

- **kiterjesztés (Broadcasting)**: eltérő, de kompatibilis alakú tenzorok elemenkénti műveleteinek kiterjesztési mechanizmusa.
- **elemenkénti művelet**: olyan művelet, amely azonos pozíciójú elemekre alkalmaz operátort.
- **skalár**: egyetlen numerikus értéket reprezentáló matematikai objektum.
- **vektor**: rendezett skalárlista (1. rendű tenzor).
- **mátrix**: kétdimenziós, sorokból és oszlopokból álló objektum (2. rendű tenzor).
- **tenzor**: többdimenziós adatszerkezet, a skalár/vektor/mátrix általánosítása.
- **eltolás (bias)**: a lineáris modell konstans tagja, amely az előrejelzés alapértékét adja.
- **torzítás (statistical bias)**: statisztikai becslők vagy modellek szisztematikus eltérése a valós értéktől.
- **gradienscsökkenés (gradient descent)**: iteratív optimalizálási eljárás, amely a veszteség csökkentése érdekében a negatív gradiens irányába frissít.
- **sztochasztikus gradienscsökkenés (SGD)**: a gradienscsökkenés véletlenszerűen kiválasztott minibatch-eken számított gradienssel.
- **veszteségfüggvény**: a modell hibáját mérő függvény, amelyet optimalizálás során minimalizálunk.
- **figyelemmechanizmus (attention mechanism)**: olyan komponens, amely a bemenet releváns részeire súlyozott fókuszt rendel.
- **önfigyelem (self-attention)**: figyelemmechanizmus, amelyben a lekérdezések, kulcsok és értékek ugyanabból a szekvenciából származnak.
- **többfejű figyelem (multi-head attention)**: párhuzamosan futtatott, különböző vetítési tereken alapuló figyelemmechanizmusok összessége.
- **pozicionális kódolás (positional encoding)**: a szekvenciában lévő elemek helyzetét kódoló reprezentáció, amelyet a figyelemmechanizmus helyzetérzékelőképességének javítására adnak hozzá a beágyazáshoz.
- **beágyazás (embedding)**: diszkrét elemek (pl. szavak) sűrű vektortérbeli reprezentációja.
- **finomhangolás (fine-tuning)**: előre betanított modell célfeladatra való továbbtanítása.
- **előtanítás (pretraining)**: modell általános reprezentáció elsajátítása nagy adathalmazon, finomhangolás előtt.
- **tanítóhalmaz (training set)**: a modell betanítására használt adathalmaz-rész.
- **validációs halmaz (validation set)**: hiperparaméter-hangoláshoz és túltanulás figyeléséhez használt adathalmaz-rész.
- **teszthalmaz (test set)**: a végső teljesítmény értékelésére szolgáló, korábban nem látott adathalmaz-rész.
- **adathalmaz (dataset)**: strukturált adatgyűjtemény, amelyből tanító/validációs/teszthalmaz képezhető.
- **regularizáció**: technika a modell általánosítóképességének javítására, amely csökkenti a túltanulást.
- **súlycsökkentés (weight decay)**: L2-regularizációs technika, amely büntet a nagy súlyokért.
- **rétegnormalizáció (layer normalization)**: normalizációs technika, amely egy adott réteg aktivációit normalizálja.
- **batchnormalizáció (batch normalization)**: normalizációs technika, amely egy batch aktivációit normalizálja.
- **generatív versengő hálózat (GAN)**: generátor és diszkriminátor versengésére épülő generatív modell.
- **értékkészletet szűkítő függvény (squashing function)**: olyan aktivációs függvény, amely széles bemeneti tartományt szűkebb kimeneti tartományba képez.
- **dimenziószám**: egy vektor vagy más objektum független komponenseinek száma.
- **maximum likelihood becslés**: statisztikai becslési módszer, amely a megfigyelt adatok likelihoodját maximalizáló paramétereket választja.
- **sajátérték / sajátvektor (eigenvalue / eigenvector)**: mátrix azon speciális skalárja/vektora, amelyre **A**v = λv teljesül.
- **sajátérték-felbontás (eigendecomposition)**: mátrix felbontása sajátértékek és sajátvektorok segítségével.
- **valószínűségi változó (random variable)**: olyan változó, amelynek értéke véletlenszerű kimenetelektől függ.
- **konfidenciaintervallum (confidence interval)**: statisztikai módszer, amely a valódi paraméterértékre ad tartásbecslést.
- **skaláris szorzat (dot product)**: két vektor elemenkénti szorzatainak összege.
- **jellemzőtérkép (feature map)**: konvolúciós réteg kimenete, amely a bemeneti tér egy adott jellemzőjére adott választ reprezentálja.
