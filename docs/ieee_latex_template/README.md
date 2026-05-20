# IEEE LaTeX Template

Esta carpeta conserva solo los archivos necesarios para compilar el informe final en `docs/term_report/main.tex`.

## Archivos necesarios

- `IEEEtran.cls`: clase LaTeX oficial usada por `\documentclass[journal,transmag]{../ieee_latex_template/IEEEtran}`.
- `IEEEtran.bst`: estilo BibTeX IEEE usado por `\bibliographystyle{../ieee_latex_template/IEEEtran}`.

## Fuente oficial

Template Selector de IEEE:

https://template-selector.ieee.org/

Selección usada:

- Publication type: `Transactions, Journals and Letters`
- Publication title: `IEEE Transactions on Magnetics`
- Article type: `Original Research`
- Format: `LaTeX for Windows or Macintosh`

ZIP LaTeX oficial:

https://template-selector.ieee.org/api/ieee-template-selector/template/504/download

ZIP de bibliografía oficial:

https://template-selector.ieee.org/api/ieee-template-selector/template/503/download

## Contenido de los ZIP oficiales

`Transactions_win_or_mac_LaTeX2e_style_file.zip` incluye:

- `IEEEtran/IEEEtran.cls`
- `IEEEtran/bare_jrnl_transmag.tex`
- `IEEEtran/bare_jrnl.tex`
- `IEEEtran/bare_jrnl_compsoc.tex`
- `IEEEtran/bare_jrnl_comsoc.tex`
- `IEEEtran/bare_conf.tex`
- `IEEEtran/bare_conf_compsoc.tex`
- `IEEEtran/bare_adv.tex`
- `IEEEtran/IEEEtran_HOWTO.pdf`
- `IEEEtran/README`
- `IEEEtran/changelog.txt`
- `Trans_Magnetics_instructions.pdf`

`Trans_Magnetics_WIN_and_MAC_bibliography_file.zip` incluye:

- `IEEEtran.bst`
- `IEEEtranN.bst`
- `IEEEtranS.bst`
- `IEEEtranSA.bst`
- `IEEEtranSN.bst`
- `IEEEabrv.bib`
- `IEEEfull.bib`
- `IEEEexample.bib`
- `IEEEtran_bst_HOWTO.pdf`
- `README`

Si se necesita consultar instrucciones, ejemplos completos o variantes del template, descargar los ZIP desde los links oficiales anteriores.

## Compilación

Desde la raíz del proyecto:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/term_report/main.tex
```

El PDF se genera como `docs/term_report/main.pdf`.
