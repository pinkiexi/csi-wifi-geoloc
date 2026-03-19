# Pipeline CSI Wi-Fi — Géolocalisation Indoor Non Supervisée

**Hanane Chahri** 

## Ce que fait ce projet


> *"automatiser la génération de données CSI à partir d'une liste d'images de plans d'habitats"*

## Pipeline
```
Générateur de plans SVG (Studio / T2 / T3 / T4)
        ↓
Parser SVG → segments de murs (format compatible CubiCasa5k)
        ↓
Simulation CSI — modèle multitrajet physique
H(f_k) = H_LOS + Σ_murs H_réfl (méthode image miroir)
        ↓
Analyse : heatmap, empreinte spectrale, PCA / t-SNE / MDS
```

## Résultats

| Figure | Contenu |
|--------|---------|
| 01_plans_generes | 12 plans variés générés automatiquement |
| 02_heatmap_csi | Amplitude CSI par position — effet des murs visible |
| 03_separation | Séparabilité des appartements (PCA/t-SNE/MDS) |
| 04_empreinte_spectrale | Signature spectrale unique par appartement |
| 05_geoloc_interne | Espace réel vs espace CSI réduit |

## Usage
```bash
pip install -r requirements.txt
python pipeline_final.py --n_plans 12 --n_subcarriers 64
```

## Limites & prochaines étapes

- Remplacer le modèle simplifié par **Sionna** pour un CSI physiquement réaliste
- Ajouter plusieurs routeurs (améliore le ratio inter/intra)
- Tester **UMAP** à la place de t-SNE
- Intégrer de vrais plans **CubiCasa5k** (même format SVG)
