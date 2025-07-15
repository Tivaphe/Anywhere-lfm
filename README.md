# Application de Chat LiquidAI avec Transformers

Une application de bureau simple pour discuter avec les modèles d'IA de LiquidAI, optimisée pour la bibliothèque `transformers`.

## ⚠️ Prérequis Important pour les Utilisateurs Windows

Pour que l'installation fonctionne, vous **DEVEZ** activer le support des chemins de fichiers longs sur votre système. C'est une opération unique et sans danger.

1.  **Ouvrez l'Éditeur de Stratégie de Groupe** :
    -   Appuyez sur `Win + R` (la touche Windows et la lettre R en même temps).
    -   Tapez `gpedit.msc` et appuyez sur Entrée.
    -   *Si cela ne fonctionne pas (versions Famille de Windows), ouvrez PowerShell en tant qu'administrateur et collez cette commande :*
        ```powershell
        New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
        ```

2.  **Naviguez dans l'Éditeur de Stratégie** (si `gpedit.msc` a fonctionné) :
    -   Allez à `Configuration ordinateur > Modèles d'administration > Système > Système de fichiers`.

3.  **Activez le support** :
    -   Dans le volet de droite, double-cliquez sur `Activer la prise en charge des chemins d'accès longs Win32`.
    -   Sélectionnez `Activé` et cliquez sur `OK`.

4.  **Redémarrez votre ordinateur**. Cette étape est cruciale pour que le changement soit pris en compte.

## Installation

### Autres Prérequis

-   **Python 3.8+**
-   **Git**

### Instructions d'Installation

#### Pour Windows

1.  Assurez-vous d'avoir suivi le prérequis important ci-dessus.
2.  Double-cliquez sur `install.bat`.

#### Pour macOS et Linux

1.  Rendez les scripts exécutables (une seule fois) : `chmod +x install.sh run.sh`
2.  Exécutez le script d'installation : `./install.sh`

## Lancement de l'Application

-   **Windows** : Double-cliquez sur `run.bat`.
-   **macOS / Linux** : Exécutez `./run.sh`.

## Comment l'utiliser

1.  **Lancez l'application**.
2.  **Sélectionnez un modèle** dans la liste.
3.  **(Optionnel) Paramètres** : Cliquez sur "Paramètres" pour ajuster le prompt système et les options de génération (température, etc.).
4.  **Discutez** !
