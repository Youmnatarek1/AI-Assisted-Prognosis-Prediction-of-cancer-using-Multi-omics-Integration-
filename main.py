import streamlit as st
import pandas as pd
import io
import subprocess
import shutil
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError
import numpy as np
import tempfile
import os
import joblib
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tempfile import NamedTemporaryFile
from rpy2.robjects import r, globalenv
import subprocess
from PIL import Image
import rpy2
import matplotlib.pyplot as plt
from io import BytesIO


# Function to load the model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        # Check if the model has the 'predict' method
        if hasattr(model, 'predict'):
            return model
        else:
            st.error("Loaded model does not have a 'predict' method.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_path = "voting.pkl"
model = pickle.load(open(model_path, 'rb'))

def predict_prognosis():
    st.markdown('<p style="color:#6495ED;font-size:28px;font-weight:bold; text-align:center;">Prognosis Prediction</p>', unsafe_allow_html=True)
    prognosis_data = st.file_uploader("Upload file for prognosis", type=['txt', 'csv', 'tsv'], accept_multiple_files=False)
    if st.button("Predict Prognosis", key='predict_btn'):
        with st.spinner("Predicting Prognosis..."):
            # Placeholder for actual input collection logic
            input_data = pd.read_csv(prognosis_data)
            ids = input_data['ids']
            X = input_data.drop(['ids'], axis = 1)
            # Impute missing values with the mean
            imputer = SimpleImputer(strategy='mean')
            X_clean = imputer.fit_transform(X)
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X_clean)

           # 0->ALive  1->DEAD
           # Perform prediction
            y = model.predict(X)
            for i, patient_id in zip(y, ids):
                if i == 1:
                    st.error(f"Patient {patient_id} is High risk")
                else:
                    st.success(f"Patient {patient_id} is Low risk")

# Function to install required R packages
def install_r_packages():
    r_executable = shutil.which("R")
    if r_executable is None:
        st.error("R is not installed or not found in your PATH. Please install R and try again.")
        return False
    else:
        packages = ['readr', 'org.Hs.eg.db', 'vsn', 'DESeq2', 'coriell', 'edgeR', 'ENmix', 'rlang']
        for package in packages:
            subprocess.run(['R', '-e', f'if (!require("{package}")) install.packages("{package}", repos="http://cran.us.r-project.org")'])
        return True

# if not install_r_packages():
#     st.stop()

# Convert R DataFrame to pandas DataFrame
pandas2ri.activate()

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding-top: 0 !important;
    }
    .sidebar .sidebar-content img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .sidebar-title {
        background-color: #E6E6FA;
        color: #6495ED;
        padding: 10px 20px;
        font-size: 24px;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .st-emotion-cache-1gv3huu {
    }
    .element-container {
        text-align: center;
    }
    .row-widget button {
        padding: 20px 40px !important;
        font-size: 18px !important;
        width: 100%;
        margin-bottom: 10px;
        background-color: white;
        color: #6495ED;  /* Set text color to blue */
        border: 1px solid #6495ED; /* Optional: Add border for better visibility */
        border-radius: 5px;
        cursor: pointer;
    }
    .row-widget button:hover {
        background-color: #B0C4DE; /* LightSteelBlue */
        color: white; /* Change text color to white on hover */
    }
    .row-widget button p {
        font-size: 18px !important;
        font-weight: bold;
    }
    .main {
        background-color: #E6E6FA;
    }
    .button-container {
        text-align: center;
        margin-top: 20px;
    }
    img {
        width: 330px !important;
        height: 260px;
        position: relative;
        top: -50px;
        left: -20px;
    }
    h1, h2, h3, p {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .content-section {
        padding: 20px;
        margin-bottom: 20px;
    }
    .content-section h2 {
        font-size: 24px;
        margin-bottom: 5rem;
        color: #4A90E2;
        text-align: center;
    }
    .content-section p {
        font-size: 18px;
        line-height: 1.5;
        text-align: justify;
        margin-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar layout with image at the top
logo_path = r"/home/el-hawary/graduation/tryme.png"  # Update this path to your logo file location
st.sidebar.image(logo_path, use_column_width=True)

# Initialize session state if not already done
if 'module' not in st.session_state:
    st.session_state.module = "Home"
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# Sidebar buttons
st.sidebar.markdown('<div class="sidebar-title">Choose Module</div>', unsafe_allow_html=True)
if st.sidebar.button("Home"):
    st.session_state.module = "Home"
if st.sidebar.button("Preprocess Data"):
    st.session_state.module = "Preprocess Data"
    st.session_state.data_type = None
    st.session_state.show_uploader = False
if st.sidebar.button("MOFA Analysis"):
    st.session_state.module = "MOFA Analysis"
    st.session_state.data_type = None
    st.session_state.show_uploader = False
if st.sidebar.button("Predict Prognosis"):
    st.session_state.module = "Predict Prognosis"
    st.session_state.data_type = None
    st.session_state.show_uploader = False
if st.sidebar.button("About Us"):
    st.session_state.module = "About Us"
    st.session_state.data_type = None
    st.session_state.show_uploader = False


if st.session_state.module == "Home":
    st.markdown(
        '<div class="content-section">'
        '<h2>Welcome to Our Data Analysis Platform</h2>'
        '<p>Data Preprocessing: Streamline your data preparation with tools tailored for RNA, miRNA, and DNA methylation datasets. Our preprocessing module allows you to filter data based on variance thresholds, ensuring clean and reliable inputs for downstream analyses.</p>'
        '<p>MOFA Analysis: Dive into Multi-Omics Factor Analysis with ease. Our platform provides the ability to define the number of factors for comprehensive data integration, helping to uncover hidden patterns and relationships in multi-omics datasets.</p>'
        '<p>Prognosis Prediction: Predict clinical outcomes with state-of-the-art machine learning models. Our prognosis prediction tools leverage your data to forecast patient prognosis, aiding in decision-making and personalized medicine strategies.</p>'
        '</div>',
        unsafe_allow_html=True
    )

elif st.session_state.module == "About Us":
    st.markdown(
        '<div class="content-section">'
        '<h2>About Us</h2>'
        '<p>The web application for our study is implemented using Streamlit, a popular open-source framework for creating data-driven web applications in Python. The front-end interface is designed to be responsive and user-friendly, providing an intuitive platform for researchers and clinicians to input multi-omics data and receive prognostic predictions. The back-end computations are powered by robust R and Python libraries tailored for multi-omics data analysis. The MOFA process is automated within this application, facilitating access to advanced analysis without requiring specialized computational expertise. Due to the complex configurations and sensitive nature of the data, local installation of the web server is not supported. Users are encouraged to utilize our public web servers for seamless and interactive analysis, or the associated R and Python packages for more flexible or batch processing.</p>'
        '</div>',
        unsafe_allow_html=True
    )

elif st.session_state.module == "Preprocess Data":
    st.markdown('<div class="content-section"><p>Select the type of data you want to preprocess. Once you choose a data type, you will be prompted to upload the data file along with the corresponding sample sheet. Ensure that the data and sample sheet are in the correct format and match in content.</p></div>', unsafe_allow_html=True)

    if not st.session_state.show_uploader:
        if st.button("RNA"):
            st.session_state.data_type = "RNA"
            st.session_state.show_uploader = True
        if st.button("miRNA"):
            st.session_state.data_type = "miRNA"
            st.session_state.show_uploader = True
        if st.button("DNA Methylation"):
            st.session_state.data_type = "DNA Methylation"
            st.session_state.show_uploader = True

# Define the R preprocessing function
def preprocess_rna(data_path, pheno_path, bad_sample_pct, bad_gene_pct, low_variance_pct, normalization_choice="Deseq"):
    data_path = data_path.replace("\\", "\\\\")
    pheno_path = pheno_path.replace("\\", "\\\\")
    r_code = f"""
    # Install and load required R packages
    
    library(readr)
    library(org.Hs.eg.db)
    library(vsn)
    library(DESeq2)
    library(coriell)
    library(edgeR)

    process_rna_data <- function(data_path, pheno_path, bad_sample_pct, bad_gene_pct, low_variance_pct, normalization_choice = "Deseq") {{
      # Read merged data
      merged_data <- read.csv(data_path, check.names = FALSE, row.names = 1, header = TRUE)
      # Filter bad samples and genes
      bad.sample <- colMeans(is.na(merged_data)) > bad_sample_pct 
      bad.gene <- rowMeans(is.na(merged_data)) > bad_gene_pct
      merged_data <- merged_data[!bad.gene, !bad.sample]
      # Load the mRNA sample sheets (metadata)
      pheno <- read_delim(pheno_path, "\\t", escape_double = FALSE, trim_ws = TRUE)
     
      # Remove low variance genes
      low_variance_pct = as.double(low_variance_pct)
      removed_mirna <- remove_var(merged_data, p = low_variance_pct)
      if (normalization_choice == "Deseq") {{
        dds <- DESeqDataSetFromMatrix(countData = removed_mirna, colData = pheno, design = ~ 1)
        dds.run <- DESeq(dds)
        x <- normTransform(dds.run)
        assy <- assay(x)
        hist(assy, main = "Histogram of Normalized RNA Data (DESeq2)")
      }} else if (normalization_choice == "edger") {{
        y <- DGEList(counts = removed_mirna)
        y <- calcNormFactors(y)
        assy <- cpm(y, normalized.lib.sizes = TRUE)
        hist(assy, main = "Histogram of Normalized RNA Data (edgeR)")
      }} else {{
        stop("Invalid normalization choice. Please choose 'Deseq' or 'edger'.")
      }}
      return(assy)
    }}
    data_path <- "{data_path}"
    pheno_path <- "{pheno_path}"
    normalized_data <- process_rna_data(data_path, pheno_path, normalization_choice = "{normalization_choice}", bad_sample_pct = "{bad_sample_pct}", bad_gene_pct= "{bad_gene_pct}", low_variance_pct = "{low_variance_pct}")
    write.csv(normalized_data, file = "normalized_rna_data.csv")
    """
    # Print R code for debugging
    #st.text(r_code)
    try:
        r(r_code)
        return "normalized_rna_data.csv"
    except RRuntimeError as e:
        st.error(f"Error occurred during R code execution: {e}")
        return None

# Streamlit app setup
def preprocess_data_rna(rna_data, rna_sample_sheet, rna_threshold, bad_sample_pct ,rna_bad_genes, normalization_choice):
    # Use a temporary directory
    temp_dir = tempfile.gettempdir()
    data_path = os.path.join(temp_dir, "rna_data.csv")
    pheno_path = os.path.join(temp_dir, "rna_sample_sheet.tsv")

    with open(data_path, "wb") as f:
        f.write(rna_data.getbuffer())
    with open(pheno_path, "wb") as f:
        f.write(rna_sample_sheet.getbuffer())

    # Read the CSV files to ensure data is numeric
    df = pd.read_csv(data_path, index_col=0)
    pheno_df = pd.read_csv(pheno_path, sep="\t")

    # Ensure only numeric columns are selected
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove low variance genes (handling done in R)
    normalized_file = preprocess_rna(data_path, pheno_path, bad_sample_pct=bad_sample_pct, bad_gene_pct=rna_bad_genes, low_variance_pct=rna_threshold, normalization_choice=normalization_choice)

    if normalized_file:
        with open(normalized_file, "rb") as f:
            return f.read()
    return None

def preprocess_mirna(data_path, pheno_path, bad_sample_pct, bad_gene_pct, low_variance_pct, normalization_choice="Deseq"):
    data_path = data_path.replace("\\", "\\\\")
    pheno_path = pheno_path.replace("\\", "\\\\")
    r_code = f"""
    # Install and load required R packages
    
    library(readr)
    library(org.Hs.eg.db)
    library(vsn)
    library(DESeq2)
    library(coriell)
    library(edgeR)

    process_rna_data <- function(data_path, pheno_path, bad_sample_pct, bad_gene_pct, low_variance_pct, normalization_choice = "Deseq") {{
      # Read merged data
      merged_data <- read.csv(data_path, check.names = FALSE, row.names = 1, header = TRUE)
      # Filter bad samples and genes
      bad.sample <- colMeans(is.na(merged_data)) > bad_sample_pct 
      bad.gene <- rowMeans(is.na(merged_data)) > bad_gene_pct
      merged_data <- merged_data[!bad.gene, !bad.sample]
      # Load the mRNA sample sheets (metadata)
      pheno <- read_delim(pheno_path, "\\t", escape_double = FALSE, trim_ws = TRUE)
     
      # Remove low variance genes
      low_variance_pct = as.double(low_variance_pct)
      removed_mirna <- remove_var(merged_data, p = low_variance_pct)
      if (normalization_choice == "Deseq") {{
        dds <- DESeqDataSetFromMatrix(countData = removed_mirna, colData = pheno, design = ~ 1)
        dds.run <- DESeq(dds)
        x <- normTransform(dds.run)
        assy <- assay(x)
        hist(assy, main = "Histogram of Normalized RNA Data (DESeq2)")
      }} else if (normalization_choice == "edger") {{
        y <- DGEList(counts = removed_mirna)
        y <- calcNormFactors(y)
        assy <- cpm(y, normalized.lib.sizes = TRUE)
        hist(assy, main = "Histogram of Normalized RNA Data (edgeR)")
      }} else {{
        stop("Invalid normalization choice. Please choose 'Deseq' or 'edger'.")
      }}
      return(assy)
    }}
    data_path <- "{data_path}"
    pheno_path <- "{pheno_path}"
    normalized_data <- process_rna_data(data_path, pheno_path, normalization_choice = "{normalization_choice}", bad_sample_pct = "{bad_sample_pct}", bad_gene_pct= "{bad_gene_pct}", low_variance_pct = "{low_variance_pct}")
    write.csv(normalized_data, file = "normalized_mirna_data.csv")
    """
    # Print R code for debugging
    #st.text(r_code)
    try:
        r(r_code)
        return "normalized_mirna_data.csv"
    except RRuntimeError as e:
        st.error(f"Error occurred during R code execution: {e}")
        return None

# Streamlit app setup
def preprocess_data_mirna(mirna_data, mirna_sample_sheet, mirna_threshold, bad_sample_pct ,mirna_bad_genes, normalization_choice):
    # Use a temporary directory
    temp_dir = tempfile.gettempdir()
    data_path = os.path.join(temp_dir, "mirna_data.csv")
    pheno_path = os.path.join(temp_dir, "mirna_sample_sheet.tsv")

    with open(data_path, "wb") as f:
        f.write(mirna_data.getbuffer())
    with open(pheno_path, "wb") as f:
        f.write(mirna_sample_sheet.getbuffer())

    # Read the CSV files to ensure data is numeric
    df = pd.read_csv(data_path, index_col=0)
    pheno_df = pd.read_csv(pheno_path, sep="\t")

    # Ensure only numeric columns are selected
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove low variance genes (handling done in R)
    normalized_file = preprocess_mirna(data_path, pheno_path, bad_sample_pct=bad_sample_pct, bad_gene_pct=mirna_bad_genes, low_variance_pct=mirna_threshold, normalization_choice=normalization_choice)

    if normalized_file:
        with open(normalized_file, "rb") as f:
            return f.read()
    return None

def preprocess_dna_methylation(data_path, bad_sample_pct, bad_gene_pct, low_variance_pct):
    r_code = f"""
    library(ENmix)
    library(readr)
    library(org.Hs.eg.db)
    library(vsn)
    library(DESeq2)
    library(coriell)
    library(edgeR)
    process_dna_methylation_data <- function(data_path, bad_sample_pct, bad_gene_pct, low_variance_pct) {{
      # Read merged data
      merged_data <- read.csv(data_path, check.names = FALSE, row.names = 1, header = TRUE)
      # Filter bad samples and genes
      bad.sample <- colMeans(is.na(merged_data)) > bad_sample_pct 
      bad.gene <- rowMeans(is.na(merged_data)) > bad_gene_pct
      merged_data <- merged_data[!bad.gene, !bad.sample]
      # Load the sample sheets (metadata)
      
     
      # Remove low variance genes
      low_variance_pct = as.double(low_variance_pct)
      removed_dna_methylation <- remove_var(merged_data, p = low_variance_pct)
      # Perform B2M
      datam = B2M(removed_dna_methylation) 
      
      hist(datam, main = "Histogram of Preprocessed DNA Methylation Data", xlab = "Methylation Levels", ylab = "Frequency")
      
      # Save the preprocessed data to a CSV file
      return(datam)
    }}

    data_path <- "{data_path}"
    
    normalized_data <- process_dna_methylation_data(data_path, bad_sample_pct = "{bad_sample_pct}",bad_gene_pct = "{bad_gene_pct}", low_variance_pct = "{low_variance_pct}" )
    write.csv(normalized_data, file = "preprocessed_meth_data.csv")
    """
    try:
        r(r_code)
        return "preprocessed_meth_data.csv"
    except RRuntimeError as e:
        st.error(f"Error occurred during R code execution: {e}")
        return None

def preprocess_data_dna_methylation(dna_methylation_data, dna_methylation_threshold, dna_methylation_bad_genes, bad_sample_pct):
    data_path = "uploaded_dna_methylation_data.csv"
    
    # Save uploaded files locally
    with open(data_path, "wb") as f:
        f.write(dna_methylation_data.getbuffer())
    
    # Call R function to preprocess DNA methylation data
    normalized_file = preprocess_dna_methylation(data_path, low_variance_pct = dna_methylation_threshold, bad_gene_pct = dna_methylation_bad_genes, bad_sample_pct = bad_sample_pct )
    
   
    if normalized_file:
        with open(normalized_file, "rb") as f:
            return f.read()
    return None


# Activate automatic conversion for pandas DataFrame to R data.frame


# # Import necessary R packages

# base = importr("base")


def run_mofa(rna_file, mirna_file, meth_file, num_factors=15, maxiter=20000, convergence_mode="slow"):
    try:
        # Read CSV files into R data.frames
        r('library("MOFA2")')
        r(f'rna_df <- read.csv("{rna_file.name}", header=TRUE, sep=",", row.names=1, check.names=FALSE)')
        r(f'mirna_df <- read.csv("{mirna_file.name}", header=TRUE, sep=",", row.names=1, check.names=FALSE)')
        r(f'meth_df <- read.csv("{meth_file.name}", header=TRUE, sep=",", row.names=1, check.names=FALSE)')
        
        # Convert DataFrames to matrices
        r("rna_matrix <- as.matrix(rna_df)")
        r("mirna_matrix <- as.matrix(mirna_df)")
        r("meth_matrix <- as.matrix(meth_df)")

        # Create a list of data frames representing the multiple views
        r("data <- list(RNA=rna_matrix, miRNA=mirna_matrix, DNA_methylation=meth_matrix)")

        # Create MOFA object
        r("MOFAobject <- create_mofa(data)")

        # Get default options
        r("data_opts <- get_default_data_options(MOFAobject)")
        r("model_opts <- get_default_model_options(MOFAobject)")
        r("train_opts <- get_default_training_options(MOFAobject)")

        # Set custom options
        r(f"model_opts$num_factors <- {num_factors}")
        r(f"train_opts$maxiter <- {maxiter}")
        r(f'train_opts$convergence_mode <- "{convergence_mode}"')

        # Prepare MOFA object
        r("MOFAobject <- prepare_mofa(object=MOFAobject, data_options=data_opts, model_options=model_opts, training_options=train_opts)")

        # Run MOFA and save the model
        outfile = f"model_{num_factors}_{convergence_mode}.hdf5"
        r(f'outfile <- "{outfile}"')
        r_code_run = """
        MOFAobject.trained <- run_mofa(MOFAobject, outfile=outfile)
        """

        # Check convergence
        r_code_check_convergence = """
        check_convergence <- function(elbo_values, n_last = 10) {
            # Extract the last n ELBO values
            last_elbo_values <- tail(elbo_values, n_last)
          
            # Check if all last n ELBO values are identical
            if (length(unique(last_elbo_values)) == 1) {
                return(TRUE)
            } else {
                return(FALSE)
            }
        }

        # Example usage:
        elbo_values <- MOFAobject.trained@training_stats$elbo

        if (check_convergence(elbo_values)) {
            convergence_message <- "The model has converged."
        } else {
            convergence_message <- "The model has not converged."
        }
        convergence_message
        """

        r(r_code_run)
        convergence_message = r(r_code_check_convergence)

        if "has converged" in convergence_message:
            st.success(convergence_message)
        else:
            st.error(convergence_message)

        return outfile, r["MOFAobject.trained"]

    except rpy2.rinterface_lib.embedded.RRuntimeError as e:
        st.error(f"An error occurred while running MOFA: {str(e)}")
        st.stop()



def plot_variance_explained(mofa_object):
    try:
        # Assign the trained MOFA object to an R variable
        r.assign("mofa_object", mofa_object)

        # Generate the plot
        r_code = """
        library(ggplot2)
        variance_plot <- plot_variance_explained(mofa_object, x='view', y='factor')
        ggsave("variance_explained_plot.png", plot=variance_plot, device="png")
        """
        r(r_code)

        # Read the saved plot and display it using matplotlib
        plot_file = BytesIO()
        with open("variance_explained_plot.png", "rb") as file:
            plot_file.write(file.read())

        plot_file.seek(0)
        img = plt.imread(plot_file)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        return fig
    except rpy2.rinterface_lib.embedded.RRuntimeError as e:
        st.error(f"An error occurred while plotting variance explained: {str(e)}")
        st.stop()


# Streamlit application for RNA
if st.session_state.data_type == "RNA":
    st.markdown('<p style="color:#6495ED;font-size:28px;font-weight:bold; text-align:center;">Upload RNA Data</p>', unsafe_allow_html=True)
    rna_data = st.file_uploader("Upload RNA Data", type=['txt', 'csv'], accept_multiple_files=False)
    rna_sample_sheet = st.file_uploader("Upload RNA Sample Sheet", type=['txt', 'csv', 'tsv'], accept_multiple_files=False)
    rna_threshold = st.slider("Select Variance Threshold for RNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    rna_bad_genes = st.slider("Select Percentage of Bad Genes to Remove for RNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    rna_bad_samples = st.slider("Select Percentage of Bad Samples to Remove for RNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    normalization_choice = st.selectbox("Select Normalization Method", ["Deseq", "edger"])
    if st.button("Run RNA Preprocessing", key='rna_preprocess_btn'):
        if rna_data is not None and rna_sample_sheet is not None:
            with st.spinner("Preprocessing RNA Data..."):
                preprocessed_rna = preprocess_data_rna(rna_data, rna_sample_sheet, rna_threshold, rna_bad_genes = rna_bad_genes, bad_sample_pct = rna_bad_samples, normalization_choice = normalization_choice)
                if preprocessed_rna:
                    st.success("RNA Data Preprocessing Completed!")
                    
                else:
                    st.error("RNA Data Preprocessing Failed!")
        else:
            st.warning("Please upload RNA data and sample sheet first.")

# Streamlit application for miRNA
if st.session_state.data_type == "miRNA":
    st.markdown('<p style="color:#6495ED;font-size:28px;font-weight:bold; text-align:center;">Upload miRNA Data</p>', unsafe_allow_html=True)
    mirna_data = st.file_uploader("Upload miRNA Data", type=['txt', 'csv'], accept_multiple_files=False)
    mirna_sample_sheet = st.file_uploader("Upload miRNA Sample Sheet", type=['txt', 'csv', 'tsv'], accept_multiple_files=False)
    mirna_threshold = st.slider("Select Variance Threshold for miRNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    mirna_bad_genes = st.slider("Select Percentage of Bad Genes to Remove for miRNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    mirna_bad_samples = st.slider("Select Percentage of Bad Samples to Remove for miRNA Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    normalization_choice = st.selectbox("Select Normalization Method", ["Deseq", "edger"])
    if st.button("Run miRNA Preprocessing", key='mirna_preprocess_btn'):
        if mirna_data is not None and mirna_sample_sheet is not None:
            with st.spinner("Preprocessing miRNA Data..."):
                preprocessed_mirna = preprocess_data_mirna(mirna_data, mirna_sample_sheet, mirna_threshold, mirna_bad_genes = mirna_bad_genes, bad_sample_pct = mirna_bad_samples, normalization_choice = normalization_choice)
                if preprocessed_mirna:
                    st.success("miRNA Data Preprocessing Completed!")
                else:
                    st.error("miRNA Data Preprocessing Failed!")
        else:
            st.warning("Please upload miRNA data and sample sheet first.")

# Streamlit application for DNA Methylation
if st.session_state.data_type == "DNA Methylation":
    st.markdown('<p style="color:#6495ED;font-size:28px;font-weight:bold; text-align:center;">Upload DNA Methylation Data</p>', unsafe_allow_html=True)
    dna_methylation_data = st.file_uploader("Upload DNA Methylation Data", type=['txt', 'csv'], accept_multiple_files=False)
    dna_methylation_threshold = st.slider("Select Variance Threshold for DNA Methylation Data", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    dna_methylation_bad_genes = st.slider("Select Percentage of Bad Genes to Remove for DNA Methylation Data",  min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    dna_methylation_bad_samples = st.slider("Select Percentage of Bad Samples to Remove for DNA Methylation Data",  min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    if st.button("Run DNA Methylation Preprocessing", key='dna_methylation_preprocess_btn'):
        if dna_methylation_data is not None:
            with st.spinner("Preprocessing DNA Methylation Data..."):
                preprocessed_dna_methylation = preprocess_data_dna_methylation(dna_methylation_data, dna_methylation_threshold = dna_methylation_threshold, dna_methylation_bad_genes = dna_methylation_bad_genes,bad_sample_pct = dna_methylation_bad_samples )
                st.success("DNA Methylation Data Preprocessing Completed!")
                
        else:
            st.warning("Please upload DNA Methylation data.")

elif st.session_state.module == "MOFA Analysis":
    st.markdown('<p style="color:#6495ED;font-size:28px;font-weight:bold; text-align:center;">MOFA Analysis</p>', unsafe_allow_html=True)
    st.header("Upload Omics Data")
    rna_file = st.file_uploader("Upload RNA CSV", type="csv")
    mirna_file = st.file_uploader("Upload miRNA CSV", type="csv")
    meth_file = st.file_uploader("Upload DNA Methylation CSV", type="csv")
  
    if rna_file and mirna_file and meth_file:
        st.success("Files uploaded successfully!")

        st.header("MOFA Parameters")

        num_factors = st.number_input("Number of Factors", value=5)
        num_factors = int(num_factors)
        if num_factors < 1:
            st.warning("Please enter a positive integer.")
        maxiter = st.number_input("Maximum Iterations", value=10)
        convergence_mode = st.selectbox("Convergence Mode", ["fast", "slow"], index=0)

        if st.button("Run MOFA"):
            trained_model, mofa_object = run_mofa(rna_file, mirna_file, meth_file,
                                                num_factors=num_factors,
                                                maxiter=maxiter,
                                                convergence_mode=convergence_mode)
            
            st.download_button("Download MOFA Model", trained_model, file_name=trained_model)

            # Plot variance explained
            st.header("Variance Explained by Factors and Views")
            fig = plot_variance_explained(mofa_object)
            st.pyplot(fig)
    
elif st.session_state.module == "Predict Prognosis":
    predict_prognosis()