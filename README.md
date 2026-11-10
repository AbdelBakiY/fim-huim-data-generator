# Synthetic Transaction Data Generator (FIM / HUIM)

Générateur de datasets transactionnels synthétiques pour évaluer les algorithmes de
**Frequent Itemset Mining (FIM)** et de **High-Utility Itemset Mining (HUIM)**.
Il comprend aussi la validation statistique, des expériences OFAT, une comparaison
d'algorithmes SPMF, une génération par LLM et une interface web Flask.

Projet réalisé dans le cadre du cours ISD 2025/2026.

Le code se trouve dans [`projet2/`](projet2/). Voir son [README](projet2/README.md) pour la documentation complète.

## Démarrage rapide

```bash
cd projet2
python -m venv .venv
# Windows : .venv\Scripts\activate    |    Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
python app.py        # puis ouvrir http://localhost:5001
```
