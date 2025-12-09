from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import re
import os
from umap import UMAP

app = Flask(__name__)

# Global variables to store data
gene_data = None
metadata = None
genes_list = []
umap_results = None
umap_df = None

def load_data():
    """Load gene expression data and metadata on startup"""
    global gene_data, metadata, genes_list, umap_results, umap_df
    
    print("Loading gene expression data...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'pariksons_gene_counts_filtered_norm.tsv')
    metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'parkinson_s.json')
    
    # Load gene expression data
    gene_data = pd.read_csv(data_path, sep='\t', index_col=0)
    genes_list = list(gene_data.index)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(genes_list)} genes and {len(gene_data.columns)} samples")
    
    # Compute UMAP
    print("Computing UMAP...")
    umap_results = UMAP(n_components=2, random_state=42).fit_transform(gene_data.T)
    
    # Create UMAP dataframe with metadata
    umap_df = pd.DataFrame(umap_results, index=gene_data.columns, columns=['UMAP_1', 'UMAP_2'])
    umap_df['sample_id'] = umap_df.index
    umap_df['tissue'] = umap_df.index.map(lambda x: normalize_tissue_source(metadata[x]['source']) if x in metadata else 'Unknown')
    umap_df['gse'] = umap_df.index.map(lambda x: metadata[x]['series'] if x in metadata else 'Unknown')
    
    print("UMAP computation complete!")

# tissue mapping I did from problem-set-3/exercise_5.py
tissue_mapping = {
    r'.*fibroblast.*|.*skin fibroblast.*': 'fibroblasts',

    r'.*ips.*|.*induced pluripotent.*': 'iPSC',
    r'.*npc.*|.*neural progenitor.*|.*neural stem.*': 'neural progenitors',

    r'.*ngn2.*|.*induced cortical neuron.*|.*iPSC derived neuron.*|.*terminally differentiated neuron.*': 'induced neurons',
    r'.*\bin neuron\b.*|.*\bin\b': 'induced dopaminergic neurons (iDANs)',

    r'.*dopaminergic.*|.*midbrain organoid.*|.*parkinson.*dopamine neuron.*': 'dopaminergic neurons',
    r'.*ventral tegmental area.*': 'brain - ventral tegmental area',

    r'.*substantia nigra.*': 'brain - substantia nigra',
    r'.*cortex.*|.*prefrontal.*|.*mtg.*|.*frontal.*': 'brain - cortex',
    r'.*amygdala.*': 'brain - amygdala',
    r'.*putamen.*': 'brain - putamen',
    r'.*temporal gyrus.*': 'brain - temporal gyrus',

    r'.*muscle.*': 'muscle',

    r'.*blood.*|.*plasma.*|.*b cell.*': 'blood',

    r'.*\bcs\d+ipd\b.*|.*gba.*|.*c2.*day5.*': 'iPD-derived cell lines',

    r'.*brain.*': 'brain (unspecified)',
}


def normalize_tissue_source(source):
    """Normalize tissue source names using the mapping from pset3 exercise_5.py"""
    clean_source = (
        source.lower()
        .replace(r'[^a-z0-9\s]', ' ')
    )
    clean_source = re.sub(r'[^a-z0-9\s]', ' ', clean_source)
    clean_source = re.sub(r'\s+', ' ', clean_source).strip()
    
    # apply mapping
    for pattern, label in tissue_mapping.items():
        if re.search(pattern, clean_source):
            return label
    
    return source


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_genes")
def get_genes():
    """Autocomplete endpoint for gene names"""
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])
    
    # find genes that start with or contain the query
    matching_genes = [gene for gene in genes_list if query in gene.upper()]

    return jsonify(matching_genes[:20])


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze gene expression across tissues"""
    gene_name = request.form.get("gene", "").strip()
    
    if not gene_name:
        return render_template("analyze.html", 
                             error="Please enter a gene name",
                             gene=gene_name)
    
    if gene_name not in genes_list:
        return render_template("analyze.html", 
                             error=f"Gene '{gene_name}' not found in dataset",
                             gene=gene_name)
    
    gene_expression = gene_data.loc[gene_name]
    
    # group by tissue type (source) with normalization
    tissue_expression = {}
    for sample_id, expression_value in gene_expression.items():
        if sample_id in metadata:
            raw_tissue = metadata[sample_id]['source']
            # Skip pancreatic islets as per exercise_5.py
            if 'pancreatic islets' in raw_tissue.lower():
                continue
            # Normalize the tissue name
            tissue = normalize_tissue_source(raw_tissue)
            if tissue not in tissue_expression:
                tissue_expression[tissue] = []
            tissue_expression[tissue].append(expression_value)
    
    # Calculate mean expression per tissue
    tissue_means = {tissue: sum(values)/len(values) 
                    for tissue, values in tissue_expression.items()}
    
    # Find tissue with highest mean expression for displaying text
    most_expressed_tissue = max(tissue_means, key=tissue_means.get)
    max_expression = tissue_means[most_expressed_tissue]
    
    fig = go.Figure()
    
    # add an interactive boxplot for each tissue
    tissues = list(tissue_expression.keys())
    for tissue in tissues:
        fig.add_trace(go.Box(
            y=tissue_expression[tissue],
            name=tissue,
            boxmean='sd',
            marker_color='lightblue'
        ))
    
    fig.update_layout(
        title={
            'text': f'Gene Expression of {gene_name} Across Different Tissues',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='Tissue Type',
        yaxis_title='Normalized Gene Expression',
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        hovermode='closest',
        xaxis={'tickangle': -45}
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Prepare analysis text
    analysis = f"The gene {gene_name} is most highly expressed in: {most_expressed_tissue}\n"
    analysis += f"Mean expression level: {max_expression:.2f}\n"
    
    return render_template("analyze.html", 
                         analysis=analysis,
                         gene=gene_name,
                         plot_html=plot_html,
                         error=None)


@app.route("/api/gene/<gene_name>")
def gene_stats(gene_name):
    """
    API endpoint to fetch expression statistics for a gene per tissue
    
    Returns JSON with statistics (mean, median, std, min, max, count) for each tissue
    """
    gene_name = gene_name.strip().upper()
    
    if not gene_name:
        return jsonify({'error': 'Gene name is required'}), 400
    
    if gene_name not in genes_list:
        return jsonify({'error': f"Gene '{gene_name}' not found in dataset"}), 404
    
    # get expression values for this gene
    gene_expression = gene_data.loc[gene_name]
    
    # group by tissue type with normalization
    tissue_expression = {}
    for sample_id, expression_value in gene_expression.items():
        if sample_id in metadata:
            raw_tissue = metadata[sample_id]['source']

            # norm tissue name
            tissue = normalize_tissue_source(raw_tissue)
            if tissue not in tissue_expression:
                tissue_expression[tissue] = []
            tissue_expression[tissue].append(expression_value)
    
    # calc statistics per tissue
    tissue_stats = {}
    for tissue, values in tissue_expression.items():
        values_array = np.array(values)
        tissue_stats[tissue] = {
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(values),
            'quartile_25': float(np.percentile(values_array, 25)),
            'quartile_75': float(np.percentile(values_array, 75))
        }
    
    sorted_tissues = sorted(tissue_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    return jsonify({
        'gene': gene_name,
        'tissues': dict(sorted_tissues),
        'total_tissues': len(tissue_stats),
        'total_samples': sum(stats['count'] for stats in tissue_stats.values())
    })


@app.route("/api/tissue_top_genes")
def tissue_top_genes():
    """API endpoint returning top and bottom N genes for a given tissue based on z-score across tissues."""
    tissue = request.args.get('tissue')
    try:
        top_n = int(request.args.get('top', 20))
    except ValueError:
        top_n = 20

    if not tissue:
        return jsonify({'error': 'tissue parameter is required'}), 400

    # Build sample -> normalized tissue mapping
    sample_tissue = {}
    for sample in gene_data.columns:
        if sample in metadata:
            sample_tissue[sample] = normalize_tissue_source(metadata[sample]['source'])
        else:
            sample_tissue[sample] = 'Unknown'

    # Create samples DataFrame and compute mean expression per tissue for each gene
    samples_df = gene_data.T.copy()
    samples_df['tissue'] = samples_df.index.map(lambda s: sample_tissue.get(s, 'Unknown'))

    # group by tissue and compute mean (result: tissues x genes), then transpose -> genes x tissues
    gene_tissue_means = samples_df.groupby('tissue').mean().T

    if tissue not in gene_tissue_means.columns:
        return jsonify({'error': f"Tissue '{tissue}' not found. Available tissues: {list(gene_tissue_means.columns)}"}), 404

    # For each gene, compute z-score of the tissue mean relative to mean/std across tissues
    across_mean = gene_tissue_means.mean(axis=1)
    across_std = gene_tissue_means.std(axis=1)

    # avoid divide-by-zero
    across_std_replaced = across_std.replace(0, np.nan)

    z_scores = (gene_tissue_means[tissue] - across_mean) / across_std_replaced
    z_scores = z_scores.fillna(0)

    df = pd.DataFrame({
        'gene': z_scores.index,
        'z': z_scores.values,
        'mean_in_tissue': gene_tissue_means[tissue].values,
        'mean_all': across_mean.values
    }).set_index('gene')

    top_df = df.sort_values('z', ascending=False).head(top_n)
    # ensure bottom list does not overlap with top list
    top_genes = set(top_df.index)
    bottom_df = df.drop(index=top_genes, errors='ignore').sort_values('z', ascending=True).head(top_n)

    top_list = [
        {
            'gene': idx,
            'z': float(row['z']),
            'mean_in_tissue': float(row['mean_in_tissue']),
            'mean_all': float(row['mean_all'])
        }
        for idx, row in top_df.iterrows()
    ]

    bottom_list = [
        {
            'gene': idx,
            'z': float(row['z']),
            'mean_in_tissue': float(row['mean_in_tissue']),
            'mean_all': float(row['mean_all'])
        }
        for idx, row in bottom_df.iterrows()
    ]

    return jsonify({'tissue': tissue, 'top': top_list, 'bottom': bottom_list})


@app.route("/tissue_genes")
def tissue_genes_page():
    """Render the tissue top/bottom genes interactive page."""
    # derive list of tissues from metadata
    tissues = set()
    for sample_id in gene_data.columns:
        if sample_id in metadata:
            tissues.add(normalize_tissue_source(metadata[sample_id]['source']))
        else:
            tissues.add('Unknown')

    tissues = sorted(list(tissues))
    return render_template('tissue_genes.html', tissues=tissues)


@app.route("/umap")
def umap_view():
    """Display interactive UMAP visualization colored by tissue type"""
    if umap_df is None:
        return render_template("umap.html", error="UMAP data not available")
    
    # create interactive plotly scatter plot
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        color='tissue',
        hover_data=['sample_id', 'gse'],
        title='UMAP Visualization of Gene Expression Data Colored by Tissue Type',
        labels={'UMAP_1': 'UMAP 1', 'UMAP_2': 'UMAP 2', 'tissue': 'Tissue Type'},
        height=700,
        width=1100
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        plot_bgcolor='white',
        title={
            'text': 'UMAP Visualization of Gene Expression Data<br><sub>Colored by Tissue Type</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#ddd',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    tissue_counts = umap_df['tissue'].value_counts().to_dict()
    
    return render_template("umap.html", 
                         plot_html=plot_html,
                         tissue_counts=tissue_counts,
                         total_samples=len(umap_df),
                         error=None)


# Load data when the app starts
load_data()

if __name__ == "__main__":
    app.run(debug=True, port=5001)