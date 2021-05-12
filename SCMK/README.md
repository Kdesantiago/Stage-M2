



1. Idée de la méthode :
=======================

Le Spectral Clustering with Multiple Kernels (SCMK) est une méthode de
clustering voulant utilisé à la fois le Spectral Clustering (SC) et la
création de matrice d’affinité afin d’améliorer les performances de
clustering.

La différence principale entre le SCMK et le Spectral Clustering est que
le SC est un algorithme en deux étapes alors que le SCMK, non. En effet,
lorsque l’on utilise le SC, on commence par construire une
représentation des données qui cherche à correspondre aux affinités
réelles, puis on applique la méthode du spectral clustering pour grouper
les individus.

Pour le SCMK, ces étapes sont faites de manière simultanée, on va
chercher à apprendre une matrice d’affinité reflétant la relation de nos
individus ainsi que la création du cluster “optimal”. Ces étapes
dépendent de paramètres qui sont mis à jour au fil de l’algorithme, et
intéragissent entre eux, permettant de profitre pleinement de
l’information générée par chacune des étapes.

2. Fonction d’objectif
======================

La fonction d’objectif que nous avons avec le SCMK est :

$$
\\min\_{Z,F,P,Q,w} \\underbrace{\\text{Tr}(K\_w - 2K\_wZ +Z^TK\_wZ) + \\alpha||Z||\_1}\_{\\text{Self-expressiveness}} + \\underbrace{\\beta\\ \\text{Tr}(P^TLP)}\_{\\text{SC - seuillage doux}} + \\underbrace{\\gamma \\ || F - PQ||^2\_F}\_{\\text{SC - seuillage dur}}
$$
 s.c. *Z* ≥ 0 (∀(*i*, *j*) *Z*<sub>*i**j*</sub> ≥ 0), diag(*Z*) = 0,
*P*<sup>*T*</sup>*P* = *I*, *Q*<sup>*T*</sup>*Q* = *I*, *F* ∈ Idx,
$K\_w = \\sum\_{i=1}^r w\_iK^i$, $\\sum\_{i=1}^r \\sqrt{w\_i}=1$,
*w*<sub>*i*</sub> ≥ 0

où Tr est la fonction Trace, et *α*, *β*, *γ* sont les paramètres de
penalités.

Notre fonction d’objectif est répartie en 3 termes + une condition de
noyau consensus :

-   Self-expressiveness :
    min<sub>*Z*</sub>Tr(*K* − 2*K**Z* + *Z*<sup>*T*</sup>*K**Z*) + *α*||*Z*||<sub>1</sub>
    s.c. *Z* ≥ 0, diag(*Z*) = 0.

-   Spectral Clustering - seuillage doux :
    min<sub>*P*</sub>*γ* Tr(*P*<sup>*T*</sup>*L**P*), s.c. *P*<sup>*T*</sup>*P* = *I*

-   Spectral Clustering - seuillage dur :
    min<sub>*F*, *Q*</sub>*γ*||*F* − *P**Q*||<sub>*F*</sub><sup>2</sup>, s.c. *Q*<sup>*T*</sup>*Q* = *I* et *F* ∈ Idx

-   Noyau consensus : $K\_w = \\sum\_{i=1}^r w\_iK^i$,
    $\\sum\_{i=1}^r \\sqrt{w\_i}=1$, *w*<sub>*i*</sub> ≥ 0

2.1. Self-expressiveness
------------------------

L’idée derrière le `self_expressiveness` est qu’une donnée peut être
reconstruit comme combinaison linéaire des autres points, et on ajoute
un terme de pénalité pour imposer une parcimonie.

A l’origine, le problème est formulé de la façon suivante :

min<sub>*Z*</sub>||*X* − *X**Z*||<sub>*F*</sub><sup>2</sup> + *α*||*Z*||<sub>1</sub>, s.c. *Z* ≥ 0, diag(*Z*) = 0

*Z* est une matrice d’affinité. Elle permet de représenter le lien entre
les données, et met un poids plus important pour des individus proches.
Le terme de pénalité permet d’avoir une représentation sparse, donnant
ainsi une représentation plus claire entre les données.

Le souci de cette formulation est qu’on suppose que tous les points se
trouvent dans une union de sous-espaces indépendants/disjoints, et que
les données sont non bruitées. Dans le cas où la structure des données
ne répond pas à ces cadres, la représentation sera parasitée, et donc
moins précise.

Pour cela, on peut généraliser la formule précédente par la suivante :

min<sub>*Z*</sub>||*ϕ*(*X*) − *ϕ*(*X*)*Z*||<sub>*F*</sub><sup>2</sup> + *α*||*Z*||<sub>1</sub>, s.c. *Z* ≥ 0, diag(*Z*) = 0
 Puis, avec un peu de travail on arrive à :
min<sub>*Z*</sub>||*ϕ*(*X*) − *ϕ*(*X*)*Z*||<sub>*F*</sub><sup>2</sup> + *α*||*Z*||<sub>1</sub>⇔ ... ⇔ min<sub>*Z*</sub>Tr(*K* − 2*K**Z* + *Z*<sup>*T*</sup>*K**Z*) + *α*||*Z*||<sub>1</sub>
 s.c. *Z* ≥ 0, diag(*Z*) = 0 et où la matrice
*K* = *ϕ*(*X*)<sup>*T*</sup>.*ϕ*(*X*) est un kernel que l’on a défini en
amont.

On arrive donc à la quantité que l’on cherche à minimiser, en *Z*,
suivante :

min<sub>*Z*</sub>Tr(*K* − 2*K**Z* + *Z*<sup>*T*</sup>*K**Z*) + *α*||*Z*||<sub>1</sub>
 s.c. *Z* ≥ 0, diag(*Z*) = 0.

Avec cette écriture, le modèle trouve les relations linéaires entre les
données dans le nouvel espace obtenu grâce au kernel, et par conséquent,
des relations non linéaires dans la représentation originale.

2.2. Spectral Clustering - seuillage doux
-----------------------------------------

Dans notre cas, nous supposons avoir c clusters. L’objectif du spectral
clustering est de trouver F tel que :
min<sub>*F* ∈ Idx</sub>Tr(*F*<sup>*T*</sup>*L**F*)
 où *F* ∈ {0, 1}<sup>*n* × *c*</sup> avec *F*<sub>*i*, *j*</sub> = 1 si
l’individu *i* est dans le cluster *j*, et 0 sinon ; *L* est le
laplacien.

On nommera F la matrice de clustering dur, car elle indique directement
dans quel groupe ce situe les individus.

Ce problème étant très complexe par la contrainte de discrétisation des
valeurs de F, on peut relaxer celle-ci pour avoir la formulation plus
habituelle du spectral clustering :

min<sub>*P*</sub>Tr(*P*<sup>*T*</sup>*L**P*), s.c. *P*<sup>*T*</sup>*P* = *I*
 où *I* représente la matrice identitée, et *P* ∈ ℝ<sup>*n* × *c*</sup>.

On nommera *P* la matrice de clusterig doux.

Par ailleurs, pour revenir à la matrice *F*, on utilise une méthode de
clustering classique comme les K-means sur *P*.

C’est par les points précédents que nous avons le terme suivant dans la
fonction d’objectif du SCMK :

min<sub>*P*</sub>*β* Tr(*P*<sup>*T*</sup>*L**P*), s.c. *P*<sup>*T*</sup>*P* = *I*

2.3. Spectral Clustering - seuillage dur
----------------------------------------

Cette partie concerne le lien entre les matrice *P* et *F*, par
l’intermédiaire d’une matrice de rotation *Q*. Si *P* est une solution
du Spectral clustering, par invariance, *P**Q* est aussi solution.
L’idée est d’avoir *P* et *Q* de telle manière que *P**Q* soit le plus
proche possible de la “vraie” matrice de clustering, la matrice de
clustering dur.

min<sub>*F*, *Q*</sub>*γ*||*F* − *P**Q*||<sub>*F*</sub><sup>2</sup>, s.c. *Q*<sup>*T*</sup>*Q* = *I* et *F* ∈ Idx

2.4 Noyau consensus
-------------------

La matrice d’affinité dépend grandement du Kernel utilisé, dont dépend
d’un choix a priori que l’on fait lors de la sélection de l’espace dans
lequel on projette nos données. Afin de rendre ce choix moins important
au sens de l’impact sur les performances de la méthode, on peut non plus
se limiter à un seul kernel mais à une combinaison convexe de ceux-ci ;
le but étant d’arriver à extraire un maximum d’information permettant de
mieux regrouper les données à partir des différents kernels.

Dans cette méthode, c’est de cette façon que les différents kernels sont
assemblés, on pondère l’information de chaque kernel en fonction de sa
pertinence, de sa contribution, pour le clustering. C’est pour cela que
dans nos autres termes nous trouvons *K*<sub>*w*</sub> le noyau
consensus, et qu’il y a les conditions suivantes dans les contraintes :

$$
K\_w = \\sum\_{i=1}^r w\_iK^i, \\sum\_{i=1}^r \\sqrt{w\_i}=1 \\text{ et } w\_i\\geq 0
$$

Par ailleurs, ici nous avons la condition que
$\\sum\_{i=1}^r \\sqrt{w\_i}=1$ car si on pose
$\\phi\_w(x) = \[\\sqrt{w\_1}\\phi\_1(x),...,\\sqrt{w\_r}\\phi\_r(x)\]$,
alors :

$$
K\_w(x,y) = &lt;\\phi\_w(x),\\phi\_w(y)&gt; = \\sum\_{i=1}^r w\_iK^i(x,y)
$$

3. Algorithme et optimisation des paramètres
============================================

Afin d’optimiser la fonction d’objectif, on va utiliser la méthode
d’optimisation Alternating Direction Method of Multipliers (ADMM) ou
Algorithme des directions alternées en français. Cette méthode utilise
le lagrangien augmentée et à la particularité de permettre, à chaque
étape d’optimisation, de fixer tous les paramètres et les optimiser un à
un.

3.1. Lagrangien augmenté
------------------------

Le problème de minimisation peut se réécrire sous la forme suivante :

Dans la norme 1 nous avons fait un changement de variable, en mettant la
matrice S à la place de la matrice Z, puis nous avons imposé une
contrainte d’égalité entre les termes. Le lagrangien augmenté qui en
découle est de la forme :
$$
{L}(S,Z,Y) = \\text{Tr}(K\_w - 2K\_wZ +Z^TK\_wZ) + \\alpha||Z||\_1 + \\beta\\ \\text{Tr}(P^TLP) + \\gamma \\ || F - PQ||^2\_F +\\frac{\\mu}{2}||S-Z+\\frac{Y}{\\mu}||^2\_F
$$

3.2. Mise à jour des paramètres
-------------------------------

### 3.2.1. S

Tout d’abord pour la matrice *S*, on souhaite la trouver telle qu’elle
correspond à :
$$
\\underset{S}{\\mathrm{argmin}} \\ \\alpha||S||\_1 + \\frac{\\mu}{2}||S-Z+\\frac{Y}{\\mu}||^2\_F
$$

On constate que l’on peut réécrire le problème sous la forme suivante :

 avec $H\_{ij} = Z\_{ij}-\\frac{Y\_{ij}}{\\mu}$ On peut donc “dériver”
cette quantité par rapport à *S*<sub>*i**j*</sub>, et annuler celle-ci.
On distingue 3 cas :

-   *S*<sub>*i**j*</sub> &gt; 0 : la dérivée de
    |*S*<sub>*i**j*</sub>| = 1, donc on a :
    $\\alpha + \\mu(S\_{ij}-H\_{ij}) = 0 \\Leftrightarrow S\_{ij} = H\_{ij} - \\frac{\\alpha}{\\mu}$
    pour $H\_{ij} &gt; \\frac{\\alpha}{\\mu}$

-   *S*<sub>*i**j*</sub> &lt; 0 : la dérivée de
    |*S*<sub>*i**j*</sub>| =  − 1, donc on a :
    $-\\alpha + \\mu(S\_{ij}-H\_{ij}) = 0 \\Leftrightarrow S\_{ij} = H\_{ij} + \\frac{\\alpha}{\\mu}$
    pour $H\_{ij} &lt; - \\frac{\\alpha}{\\mu}$

-   *S*<sub>*i**j*</sub> = 0 : Les sous-gradients de la norme 1 varient
    entre \]-1,1\[ mais la valeur de *S*<sub>*i**j*</sub> reste à 0.

Donc, on met à jour S par l’équation :

### 3.2.2. Z

On commence par poser des notations :

-   $E = S + \\frac{Y}{\\mu}$

-   *D* tel que
    *D*<sub>*i*, *j*</sub> = ||*P*<sub>*i*, :</sub> − *P*<sub>*j*, :</sub>||<sub>2</sub><sup>2</sup>

-   $F(Z) = \\text{Tr}(K\_w - 2K\_wZ +Z^TK\_wZ) + \\beta \\ \\text{Tr}(P^TLP) + \\frac{\\mu}{2}||S-Z+\\frac{Y}{\\mu}||^2\_F$

Par ailleurs, on remarquera que
$\\sum\_{ij}\\frac{1}{2}||P\_{i,:} - P\_{j,:}||^2\_2 s\_{ij} = \\text{Tr}(P^TLP)$.

On cherche donc :

On pose donc :
$\\tilde{F}(Z) = \\text{Tr}(- 2K\_wZ +Z^TK\_wZ) + \\frac{\\beta}{2} \\ Tr(DZ) + \\frac{\\mu}{2}(-2Tr(E^TZ) + Tr(Z^TZ))$,
et on cherche donc à annuler la dérivée partielle en Z de cette
fonction.

### 3.2.3. Y

*Y* est le multiplicateur de lagrange associé au problème. Ce paramètre
gère le lien entre les matrices *S* et *Z* et évolue selon celui-ci. En
effet, plus *S* et *Z* seront éloignées, plus les valeurs dans *Y*
seront grandes, et inversement. De plus, lors de la mise à jour de *Y*
un paramètre d’apprentissage, de pénalisation, *μ* intervient,
permettant d’ajuster le poids de la contrainte d’égalité.

### 3.2.4. P

Il s’agit de la variable la plus compliquée à optimiser de l’algorithme.
On cherche donc la matrice *P* telle que :
$$
\\underset{P}{\\mathrm{argmin}}\\ \\text{Tr}(P^TLP) + \\gamma \\ || F - PQ||^2\_F , \\text{ s.c. }P^TP=I
$$

Si on s’occupait que de trouver
$\\underset{P}{\\mathrm{argmin}}\\ \\text{Tr}(P^TLP), \\text{ s.c. }P^TP=I$,
on serait dans la démarche du spectral clustering. Pour trouver le *P*
optimal, il nous suffirait de faire la décomposition en valeurs
singulières de *L*, puis de prendre les *c* dernières valeurs propres,
et enfin normaliser chaque ligne.

A FINIR

### 3.2.5. Q

Pour être à jour la matrice *Q*, nous devons résoudre le problème
$$
\\underset{Q}{\\mathrm{argmin}}\\ \\gamma \\ || F - PQ||^2\_F, \\text{ s.c. } Q^TQ=I
$$
 Ceci est un problème de Procruste orthogonal, avec pour matrice cible
*F* et la matrice à transformer *P*. Afin de le résoudre, il nous suffit
de suivre la démarche présentée dans l’article consacré (disponible ici
: [**lien**](https://link.springer.com/article/10.1007/BF02289451)).

Tout d’abord, il nous faut faire la décomposition en valeurs singulières
de la matrice *M* = *F**Y*<sup>*T*</sup>, nous donnant
*M* = *U**Σ**V*<sup>*T*</sup>. Ensuite, on obtient *Q* par simple
produit des parties gauche et droite de la décomposition :

### 3.2.6. F

Notre objectif est de trouver la matrice *F* telle qu’elle corresponde à
:
$$
\\underset{F}{\\mathrm{argmin}} \\ \\gamma \\ || F - PQ||^2\_F, \\text{ s.c. }  F \\in \\text{Idx}
$$

Pour maximiser cette quantité, s.c. *F* ∈ Idx, on constate que l’on doit
construire F telle que :

### 3.2.7. w

On cherche
$$
\\underset{w}{\\mathrm{argmin}} \\sum\_{i=1}^r w\_ih\_i , \\text{ s.c. } \\sum\_{i=1}^r \\sqrt{w\_i}=1, w\_i\\geq 0
$$
 avec
*h*<sub>*i*</sub> = Tr(*K*<sup>*i*</sup> − 2*K*<sup>*i*</sup>*Z* + *Z*<sup>*T*</sup>*K*<sup>*i*</sup>*Z*),
où *K*<sup>*i*</sup> représente le i-ème kernel.

On peut écrire le lagrangien du problème :

$$
J(w) = w^Th + \\xi (1- \\sum\_{i=1}^r \\sqrt{w\_i})
$$

Par les conditions de Karush-Kuhn-Tucker (KKT), on veut que
$\\xi (1- \\sum\_{i=1}^r \\sqrt{w\_i}) = 0$, on suppose aussi que *γ*
est différent de 0. On arrive donc à l’équation :

On revient à *w*<sub>*i*</sub>

3.3. Algorithme
---------------

4. Exemple(s)
=============

4.1. 3 Gaussiennes
------------------

4.2. Autre exemple plus compliqué (1 noyau puis mélange)
--------------------------------------------------------

Autre version :
===============

Remplacer le fait de trouver Q et F, par un K-means sur P.

5. Discussion
=============

-   Temps de calcul
-   Optimisation hyper-paramètres
-   Souci de stabilité

Conclusion
==========
