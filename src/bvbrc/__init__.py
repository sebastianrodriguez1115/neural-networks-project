"""
bvbrc — Cliente para la API REST de BV-BRC.

Exporta las funciones públicas de los módulos amr y genomes
para que se puedan importar directamente desde el paquete:

    from bvbrc import fetch_amr_labels, download_multiple_genomes_fasta
"""

from .amr import ESKAPE_TAXON_IDS, fetch_amr_labels
from .genomes import download_genome_fasta, download_multiple_genomes_fasta

__all__ = [
    "ESKAPE_TAXON_IDS",
    "fetch_amr_labels",
    "download_genome_fasta",
    "download_multiple_genomes_fasta",
]
