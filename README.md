# Hands-On Machine Learning

A study workspace for working through [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) by Aurélien Géron.

Jupyter notebooks with notes, experiments, and exercises — organized by chapter.

## Setup

```bash
# Create and activate virtual environment (Python 3.11+)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## Project Structure

```
notebooks/
  01_the_machine_learning_landscape.ipynb
  02_end_to_end_machine_learning_project.ipynb
  datasets/                  ← downloaded automatically on first run

Concepts.md                  ← detailed explanations with diagrams (by chapter phase)
ML Glossary.md               ← quick-reference lookup for ML terms

requirements.txt             ← pinned dependencies
```

## Key Libraries

| Category | Libraries |
|----------|-----------|
| ML / Data Science | scikit-learn, pandas, numpy, scipy |
| Deep Learning | PyTorch (torch, torchvision, torchaudio) |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Lab |

## Notes

- Notebooks download datasets on first run (California housing data from `github.com/ageron/data`)
- Version requirements: Python >= 3.10, scikit-learn >= 1.6.1
- [Concepts.md](Concepts.md) follows the notebook's 5-phase structure — scroll to the phase you're working on
- [ML Glossary.md](ML%20Glossary.md) is a living glossary of terms encountered in each chapter
