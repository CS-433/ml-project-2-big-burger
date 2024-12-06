# Okayy la team voila ce qu'on doit faire:

- faire un model CNN comme le paper (mathis)
- faire un model Unet (anoush)
- faire un model ResNet (emilien)

- faire la pipeline qui génère les images, entraine tous les modèles en même temps (emilien)




Faire de la Verification: 
- Avoir un test/validation set d'images qu'on peut générer une fois et qu'on save, auquel on compare nos modèles à chaque ieration/epoque -> générer que les positions et les stocker, et regénérer les images après 
- Calculer le coarse D
- changer le code pipelie, sortir le optimizer et modifier le fonctionnement d'époque
- Utiliser des D plus petits (entre 0 et 10) diviser pour entrainer 
- Centering et entrainer un modèle plus petit 
- Diminuer bruit, diminuer pixel size, diminuer background 
- run multicore, optimizer code génération



## 6/12:

- Exclure les D trop grands (particule qui sort): Out FOV
- Plot le D frame-to-frame sur (coarse  D)
- Pour utiliser la vraie data, entrainer nouveaux modeles avec en input 64x64x16

Réflexions par rapport à l'architecture du papier, comparer avec la notre
Beaucoup de 

Quest-ce qui est appris par le modèle (shape ou total displacement ou intensité des peaks) ?


Architectures:
Augmeneter la taille du Resnet -> conv2D vs conv3D
Tester architectures papers différentes

Metrics: 
R^2, % error, Mean, pearson corr
