# BV-BRC API — Referencia para implementación

> Investigación realizada 2026-03-03 para implementar `src/bvbrc_client.py`.

## Base URL

```
https://www.bv-brc.org/api/
```

Documentación oficial: https://www.bv-brc.org/api/doc/

---

## Autenticación

Los datos públicos **no requieren autenticación**. Los datos de ESKAPE en BV-BRC son públicos.

---

## Lenguaje de consulta: RQL

Las queries usan **Resource Query Language (RQL)**:

| Operador | Significado | Ejemplo |
|---|---|---|
| `eq(field,value)` | igualdad | `eq(taxon_id,1280)` |
| `in(field,(v1,v2,...))` | pertenencia | `in(taxon_id,(1280,573))` |
| `ne(field,value)` | distinto | `ne(resistant_phenotype,Intermediate)` |
| `and(expr1,expr2)` | AND lógico | `and(eq(taxon_id,573),eq(evidence,Laboratory))` |
| `select(f1,f2,...)` | proyección | `select(genome_id,antibiotic)` |
| `sort(+field)` | orden asc | `sort(+genome_id)` |
| `limit(N,offset)` | paginación | `limit(25000,0)` |

Parámetros especiales:
- `http_accept=application/json` — formato de respuesta
- `http_download=true` — sube el límite a 25M registros (requiere `sort`)

---

## ESKAPE: taxon IDs

| Organismo | taxon_id |
|---|---|
| *Enterococcus faecium* | 1352 |
| *Staphylococcus aureus* | 1280 |
| *Klebsiella pneumoniae* | 573 |
| *Acinetobacter baumannii* | 470 |
| *Pseudomonas aeruginosa* | 287 |
| *Enterobacter* spp. (género) | 547 |

Para *Enterobacter* a nivel de género usar `taxon_lineage_ids` en lugar de `taxon_id`:
```
eq(taxon_lineage_ids,547)
```

---

## Endpoints principales

### 1. Genomas — colección `genome`

```
GET https://www.bv-brc.org/api/genome/?{RQL_QUERY}
```

Campos útiles: `genome_id`, `taxon_id`, `organism_name`, `genome_status`

**Ejemplo — todos los genomas ESKAPE:**
```
https://www.bv-brc.org/api/genome/?in(taxon_id,(1352,1280,573,470,287,547))
  &select(genome_id,taxon_id,organism_name,genome_status)
  &limit(25000,0)&sort(+genome_id)&http_download=true&http_accept=text/tsv
```

---

### 2. Etiquetas AMR — colección `genome_amr`

```
GET https://www.bv-brc.org/api/genome_amr/?{RQL_QUERY}
```

Campos clave:

| Campo | Descripción |
|---|---|
| `genome_id` | FK a colección `genome` |
| `taxon_id` | ID taxonómico NCBI |
| `antibiotic` | Nombre del antibiótico (minúsculas) |
| `resistant_phenotype` | `"Resistant"`, `"Susceptible"`, `"Intermediate"`, `"Non-susceptible"` |
| `evidence` | `"Laboratory"` o `"Computational"` |
| `laboratory_typing_method` | `"MIC"`, `"disk diffusion"`, `"agar dilution"`, etc. |
| `measurement` | String MIC completo, e.g. `"<=0.5"` |
| `measurement_sign` | `"<"`, `"<="`, `"="`, `">="`, `">"` |
| `measurement_value` | Parte numérica del MIC |
| `measurement_unit` | `"mg/L"` o `"mm"` |
| `testing_standard` | `"CLSI"`, `"EUCAST"`, etc. |

**Ejemplo — AMR ESKAPE con evidencia de laboratorio, excluyendo Intermediate:**
```
https://www.bv-brc.org/api/genome_amr/
  ?and(in(taxon_id,(1352,1280,573,470,287,547)),
       ne(resistant_phenotype,Intermediate),
       eq(evidence,Laboratory))
  &select(genome_id,taxon_id,antibiotic,resistant_phenotype,laboratory_typing_method,testing_standard)
  &limit(25000,0)&sort(+genome_id)&http_download=true&http_accept=text/tsv
```

**Alternativa bulk (FTP):** El archivo `ftps://ftp.bvbrc.org/RELEASE_NOTES/PATRIC_genomes_AMR.txt` contiene todos los registros AMR en un solo TSV. Es la opción más eficiente para la descarga inicial.

---

### 3. Secuencias FASTA — colección `genome_sequence`

```
GET https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{GID})&http_accept=application/dna+fasta
```

Devuelve el FASTA de todos los contigs del genoma.

**FTP alternativo** (puede tener problemas de acceso — verificar):
```
ftps://ftp.bvbrc.org/genomes/{genome_id}/{genome_id}.fna
```

---

## Paginación

La respuesta incluye el header `Content-Range`:
```
Content-Range: items 0-24999/153422
```
Usar el total para decidir si continuar paginando. Límite por request: 25,000 registros (JSON/TSV normal).

---

## Recomendaciones de implementación

1. **Etiquetas AMR primero:** descargar `PATRIC_genomes_AMR.txt` vía FTP o la query bulk via API → obtener lista de `genome_id` con etiquetas válidas
2. **Descargar FASTA solo** de genomas que tengan al menos una etiqueta R/S en el dataset — evita descargar miles de genomas innecesarios
3. **Paginación:** usar `limit(25000, offset)` + `sort(+genome_id)` + `Content-Range` header
4. **Rate limiting:** no hay límite documentado; agregar `time.sleep(0.5)` entre llamadas por-genoma como cortesía
5. **POST para queries largas:** usar POST en vez de GET si el query string supera límites de URL

---

## SDK / herramientas de referencia

No existe SDK oficial en Python. Referencias útiles:
- BV-BRC CLI (`p3-*`, Perl): https://www.bv-brc.org/docs/cli_tutorial/
- JS client (referencia de implementación): https://github.com/BV-BRC/bvbrc_js_client
- QIIME2 RESCRIPt plugin (wrapper Python más cercano): https://docs.qiime2.org/2024.10/plugins/available/rescript/get-bv-brc-genomes/
