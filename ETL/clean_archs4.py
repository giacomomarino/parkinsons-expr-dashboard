
import os
import qnorm
import json
from umap import UMAP
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# Searched GEO metadata for parkinson(?:'s)? disease related samples on https://archs4.org/data on Oct 10, 2025
# and downloaded the gene counts and metadata

# This comprises 606 samples (GSMs) from 46 studies (GSEs)

pariksons_gene_counts = pd.read_csv(f'data/parkinson_gene_counts.tsv', sep='\t', index_col=0)

with open(f'data/parkinson_s.json', 'r') as f:
    geo_metadata = json.load(f)

meta_df = pd.DataFrame.from_dict(geo_metadata, orient='index')

pariksons_gene_counts.sum(axis=0).hist(bins=100)
plt.xlabel('Number of reads')
plt.ylabel('Frequency (# samples)')
plt.savefig('out/pariksons_gene_counts_histogram.png', bbox_inches='tight')
plt.show()


#%%


meta_df.value_counts('series', ascending=True).plot.barh(figsize=(5, 8))
plt.xlabel('Number of samples')
plt.savefig('out/pariksons_gene_counts_histogram_num_samples_per_gse.png', bbox_inches='tight')
plt.show()

#%%
UMAP_results = UMAP(n_components=2).fit_transform(pariksons_gene_counts.T)

umap_df = pd.DataFrame(UMAP_results, index=pariksons_gene_counts.columns, columns=['UMAP_1', 'UMAP_2'])
umap_df['num_reads'] = pariksons_gene_counts.sum(axis=0)
umap_df['GSE'] = umap_df.index.map(lambda x: geo_metadata[x]['series'])
umap_df['source'] = umap_df.index.map(lambda x: geo_metadata[x]['source'])
umap_df['characteristics'] = umap_df.index.map(lambda x: geo_metadata[x]['characteristics'])

plt.figure(figsize=(8, 7))
for gse in umap_df['GSE'].unique():
    umap_df_gse = umap_df[umap_df['GSE'] == gse]
    plt.scatter(umap_df_gse['UMAP_1'], umap_df_gse['UMAP_2'], s=2, label=gse)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.yticks([])
plt.xticks([])
plt.savefig('out/pariksons_umap_unnormalized_gse.png', bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 7))
plt.scatter(umap_df['UMAP_1'], umap_df['UMAP_2'], s=2, c=np.log(umap_df['num_reads'] + 1), cmap='viridis')
plt.colorbar(label='log(Number of reads)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.yticks([])
plt.xticks([])
plt.savefig('out/pariksons_umap_unnormalized_reads.png', bbox_inches='tight')
plt.show()


meta_df.value_counts('source', ascending=True).plot.barh(figsize=(5, 14))
plt.xlabel('Number of samples')
plt.ylabel('Source')
plt.savefig('out/pariksons_gene_counts_histogram_num_samples_per_source.png', bbox_inches='tight')
plt.show()


# %%
###### Normalization ######


## we may also want to filter some of the samples by the source/ sort by tissue cell type profiled
# Lots of different sources, mostly brain and neuronal cells
# Needs lot of normalization

## strangely there are 124 samples in pancreatic islets
## looking at the metadata more closely they have this in the
# characteristics: "cryopreserved: No,diabetes: Other_Parkinson,Sex: M,patched: Yes"
## so these are artifacts and we can filter them out
meta_df_filtered = meta_df[meta_df['source'] != 'pancreatic islets']
gene_counts_filtered = pariksons_gene_counts[meta_df_filtered.index]


meta_df_filtered['clean_source'] = (
    meta_df_filtered['source']
    .str.lower()
    .str.replace(r'[^a-z0-9\s]', ' ', regex=True)  # remove punctuation
    .str.replace(r'\s+', ' ', regex=True)          # collapse spaces
    .str.strip()
)

mapping = {
    # === Fibroblasts and skin ===
    r'.*fibroblast.*|.*skin fibroblast.*': 'fibroblasts',

    # === iPSC and derivatives ===
    r'.*ips.*|.*induced pluripotent.*': 'iPSC',
    r'.*npc.*|.*neural progenitor.*|.*neural stem.*': 'neural progenitors',

    # === Induced neurons and neuronal cultures ===
    r'.*ngn2.*|.*induced cortical neuron.*|.*iPSC derived neuron.*|.*terminally differentiated neuron.*': 'induced neurons',
    r'.*\bin neuron\b.*|.*\bin\b': 'induced dopaminergic neurons (iDANs)',  # captures "iN" or "induced neuron"

    # === Dopaminergic / midbrain ===
    r'.*dopaminergic.*|.*midbrain organoid.*|.*parkinson.*dopamine neuron.*': 'dopaminergic neurons',
    r'.*ventral tegmental area.*': 'brain - ventral tegmental area',

    # === Brain subregions ===
    r'.*substantia nigra.*': 'brain - substantia nigra',
    r'.*cortex.*|.*prefrontal.*|.*mtg.*|.*frontal.*': 'brain - cortex',
    r'.*amygdala.*': 'brain - amygdala',
    r'.*putamen.*': 'brain - putamen',
    r'.*temporal gyrus.*': 'brain - temporal gyrus',

    # === Muscle ===
    r'.*muscle.*': 'muscle',

    # === Blood and plasma ===
    r'.*blood.*|.*plasma.*|.*b cell.*': 'blood',

    # === Experimental dataset tags (iPD, CS, C2, GBA) ===
    r'.*\bcs\d+ipd\b.*|.*gba.*|.*c2.*day5.*': 'iPD-derived cell lines',

    # === Generic brain ===
    r'.*brain.*': 'brain (unspecified)',
}


def normalize_source(text):
    for pattern, label in mapping.items():
        if re.search(pattern, text):
            return label
    print(text)
    return text

meta_df_filtered['normalized_source'] = meta_df_filtered['clean_source'].apply(normalize_source)
meta_df_filtered.value_counts('normalized_source', ascending=True).plot.barh()
plt.savefig('out/pariksons_gene_counts_filtered_source_normalized.png', bbox_inches='tight')
plt.show()

# %%


norm_exp = qnorm.quantile_normalize(np.log2(1+np.array(gene_counts_filtered)))
gene_counts_filtered_norm = pd.DataFrame(norm_exp, index=gene_counts_filtered.index, columns=gene_counts_filtered.columns, dtype=np.float32)
os.makedirs('data', exist_ok=True)
gene_counts_filtered_norm.to_csv('data/parkinson_gene_counts_filtered_norm.tsv', sep='\t')


print("After normalization log-quantile normalization:")
gene_counts_filtered_norm.sum(axis=0).mean(), gene_counts_filtered_norm.sum(axis=0).std()

#%%
UMAP_norm = UMAP(n_components=2).fit_transform(gene_counts_filtered_norm.T)

umap_df_norm = pd.DataFrame(UMAP_norm, index=gene_counts_filtered_norm.columns, columns=['UMAP_1', 'UMAP_2'])
umap_df_norm['num_reads'] = gene_counts_filtered_norm.sum(axis=0)
umap_df_norm['GSE'] = umap_df_norm.index.map(lambda x: geo_metadata[x]['series'])
umap_df_norm['normalized_source'] = umap_df_norm.index.map(lambda x: meta_df_filtered.loc[x]['normalized_source'])



plt.figure(figsize=(8, 7))
plt.scatter(umap_df_norm['UMAP_1'], umap_df_norm['UMAP_2'], s=2, c=np.log(umap_df_norm['num_reads'] + 1), cmap='viridis')
plt.colorbar(label='log Number of reads')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.yticks([])
plt.xticks([])
plt.savefig('out/pariksons_umap_qnorm_norm_reads.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 7))
for gse in umap_df_norm['GSE'].unique():
    umap_df_norm_gse = umap_df_norm[umap_df_norm['GSE'] == gse]
    plt.scatter(umap_df_norm_gse['UMAP_1'], umap_df_norm_gse['UMAP_2'], s=3, label=gse)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.yticks([])
plt.xticks([])
plt.savefig('out/pariksons_umap_qnorm_gse.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 7))
for source in umap_df_norm['normalized_source'].unique():
    umap_df_norm_gse = umap_df_norm[umap_df_norm['normalized_source'] == source]
    if len(umap_df_norm_gse) > 1:
        plt.scatter(umap_df_norm_gse['UMAP_1'], umap_df_norm_gse['UMAP_2'], s=3, label=source)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.yticks([])
plt.xticks([])
plt.savefig('out/pariksons_umap_qnorm_norm_source.png', bbox_inches='tight')
plt.show()