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
