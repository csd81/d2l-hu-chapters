# A Google Colab használata
:label:`sec_colab`

A :numref:`sec_sagemaker` és :numref:`sec_aws` szakaszokban bemutattuk, hogyan futtatható ez a könyv AWS-en. Egy másik lehetőség a könyv futtatása a [Google Colab](https://colab.research.google.com/) platformon, ha van Google-fiókunk.

Egy szakasz kódjának Colabban való futtatásához egyszerűen kattintsunk a `Colab` gombra, ahogyan az :numref:`fig_colab` ábrán látható.

![Egy szakasz kódjának futtatása Colabon.](../img/colab.png)
:width:`300px`
:label:`fig_colab`


Ha először futtatunk egy kódcellát,
egy figyelmeztető üzenetet kapunk, ahogyan az :numref:`fig_colab2` ábrán látható.
A figyelmeztetés figyelmen kívül hagyásához kattintsunk a „RUN ANYWAY" gombra.

![A figyelmeztető üzenet figyelmen kívül hagyása a „RUN ANYWAY" gombra kattintva.](../img/colab-2.png)
:width:`300px`
:label:`fig_colab2`

Ezután a Colab egy olyan példányhoz csatlakozik, amelyen az adott szakasz kódja futtatható.
Konkrétan, ha GPU-ra van szükség,
a Colab automatikusan
egy GPU-példányhoz csatlakozik.


## Összefoglalás

* A Google Colab segítségével a könyv bármely szakaszának kódja futtatható.
* Ha a könyv valamely szakaszához GPU szükséges, a Colab automatikusan GPU-példányhoz csatlakozik.


## Gyakorlatok

1. Nyissuk meg a könyv bármely szakaszát a Google Colab segítségével.
1. Szerkesszünk és futtassunk egy GPU-t igénylő szakaszt a Google Colab segítségével.


[Megbeszélések](https://discuss.d2l.ai/t/424)
