import pandas as pd


def if_mutation(mu_data, cell_line_id, gene_id):
    filtered_rows = mu_data[mu_data['ModelID'] == cell_line_id]
    # mu_event = filtered_rows[filtered_rows.iloc[:, -1].astype(str).isin([gene_id, f"{gene_id}.0"])]
    mu_event = filtered_rows[filtered_rows['EnsemblGeneID'].astype(str) == gene_id]
    
    return (1 if not mu_event.empty else 0)


def get_omics_data(ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id):
    ge_level = None
    cnv_gene = None

    # Gene Expression (GE)
    try:
        gene_columns = [col for col in ge_data.columns if gene_name in col]
        matched_column = gene_columns[0] 
        ge_level = ge_data.loc[cell_line_id, matched_column]
    except (IndexError, KeyError):
        ge_level = None 
    
    # Copy Number Variation (CNV)
    try:
        gene_columns = [col for col in cn_gene_data.columns if gene_name in col]
        matched_column = gene_columns[0]
        cnv_gene = cn_gene_data.loc[cell_line_id, matched_column]
        if pd.isna(cnv_gene):
            cnv_gene = None
    except (IndexError, KeyError):
        cnv_gene = None

    # MU
    mu_gene = if_mutation(mu_data, cell_line_id, gene_id)

    return [ge_level, cnv_gene, mu_gene]

def format_cell_line_description(cell_line, gene_name, omics_data):

    ge_level, cnv_gene, mu_gene = omics_data
    mutation_status = "mutated" if mu_gene == 1 else "not mutated"
    template = (
        f"In the {cell_line} cell line, {gene_name} has a "
        f"gene expression level of {ge_level}, "
        f"a copy number variation of {cnv_gene}, "
        f"and is {mutation_status}."
    )
    return template

