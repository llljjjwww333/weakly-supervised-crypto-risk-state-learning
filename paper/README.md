# Paper Source

This folder contains the KBS submission-style manuscript source and the compiled PDF snapshot.

Included:

- `Manuscript.tex`
- `Manuscript.pdf`
- bibliography and CAS class/style files required for compilation

Compile with:

```bash
xelatex Manuscript.tex
bibtex Manuscript
xelatex Manuscript.tex
xelatex Manuscript.tex
```

Figures are resolved from `../figures/main/`.
