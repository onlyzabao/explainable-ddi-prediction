import colorsys
import streamlit as st
import pandas as pd
import logging
import time
import subprocess
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
import tempfile
from pyvis.network import Network
import os

HETIONET_COLORS = {
    'Gene': '#ff4444',         # Bright red
    'Compound': '#ff8c00',     # Dark orange
    'Anatomy': '#9467bd',      # Purple
    'Disease': '#2ecc71',      # Emerald green
    'Symptom': '#e74c3c',      # Darker red
    'Side Effect': '#e67e22',  # Orange
    'Biological Process': '#3498db',  # Blue
    'Cellular Component': '#1abc9c',  # Turquoise
    'Molecular Function': '#9b59b6',  # Purple
    'Pathway': '#f1c40f',      # Yellow
    'Pharmacologic Class': '#34495e'  # Dark blue-gray
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@st.cache_data
def load_drugs():
    """Load and cache drug data"""
    drugs = pd.read_csv("../../data/resources/pharmacokkinetic_drugs.txt", header=None)
    drugs.set_index(0, inplace=True)
    logging.info("Drug data loaded successfully.")
    return drugs

def finalize_predicion(src_drug, query_relation):
    """Finalize the prediction results"""
    with open("../../results/test/CiC", "r") as file:
        answers = file.read().split("___\n")

    rows = []
    for answer in answers:
        lines = answer.strip().split("\n")
        if len(lines) < 4:
            continue

        idx = 2 if len(lines) > 4 else 0
        instance_path = lines[idx].split("\t")
        meta_path = lines[idx + 1].split("\t")

        if instance_path[3].split("::")[0] != "Compound":
            continue

        rows.append([
            f"{instance_path[0]}\t{meta_path[0]}",
            f"{instance_path[1]}\t{meta_path[1]}",
            f"{instance_path[2]}\t{meta_path[2]}",
            instance_path[3]
        ])

    explanations = pd.DataFrame(rows)

    predictions = explanations.iloc[:, 3].value_counts().reset_index()
    predictions.columns = ['Target Drug', 'Reasoning Paths']

    ddi = pd.read_csv("../../data/resources/pharmacokinetic_ddi.tsv", sep="\t")
    ddi_filtered = ddi[ddi["ID1"] == src_drug]

    predictions = predictions.merge(ddi_filtered[['ID2', 'Map']], left_on='Target Drug', right_on='ID2', how='left')

    predictions.sort_values(by="Map", ascending=False, inplace=True)
    predictions.reset_index(drop=True, inplace=True)
    predictions = predictions[['Target Drug', 'Reasoning Paths', 'Map']]
    predictions.rename(columns={'Map': 'Known Interaction'}, inplace=True)
    predictions['Known Interaction'] = predictions['Known Interaction'].str.replace("#Drug2", "Source Drug").str.replace("#Drug1", "Target Drug")
    predictions['Target Drug'] = predictions['Target Drug'].str.replace("Compound::", "")

    return predictions, explanations

def predict_ddi(search_input):
    """Run the DDI prediction pipeline"""
    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        # Set up
        progress_text.write("Step 0: Setting up environment...")
        time.sleep(1)
        src_drug = "Compound::" + search_input
        query_relation = "CiC"
        tgt_drug = "PAD"
        
        with open("../../data/datasets/test.txt", "w") as file:
            file.write(f"{src_drug}\t{query_relation}\t{tgt_drug}\n")

        progress_bar.progress(20)

        # Run prediction
        progress_text.write("Step 1: Running prediction command...")
        command = "cd .. && ./run.sh ../configs/test_hetionet_4+1_p3_e1.sh"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for idx, line in enumerate(iter(process.stdout.readline, "")):
            line = line.strip().replace("INFO:__main__:", "")
            logging.info(line)
            progress_text.write(line)
            time.sleep(1)
            progress_bar.progress(min(20 + idx * 10, 90))
        
        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        # Visualize results
        progress_text.write("Step 3: Visualizing results...")
        predictions, explanations = finalize_predicion(src_drug, query_relation)
        progress_bar.progress(100)

        return predictions, explanations

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with error: {e}")
        st.error(f"An error occurred while running the prediction command:\n{e}")
        return None
    except Exception as e:
        logging.error(f"Error during prediction for {search_input}: {e}")
        st.error("An unexpected error occurred during prediction. Please check the log.")
        return None

def get_node_color(node_type):
    """Get color for a node type"""
    return HETIONET_COLORS.get(node_type, '#7f8c8d')  # Default gray for unknown types

def construct_graph(data):
    """Construct a graph from the data"""
    edges = []
    start_nodes = set()
    end_nodes = set()
    middle_nodes = set()
    
    for i, row in data.iterrows():
        line = "\t".join(row)
        parts = line.split("\t")
        
        # Add start and end nodes
        start_nodes.add(parts[0])
        end_nodes.add(parts[-1])
        
        # Add middle nodes and edges
        for i in range(0, len(parts)-1, 2):
            source = parts[i]
            relation = parts[i+1]
            target = parts[i+2]
            
            # Handle reverse relationships
            if relation.startswith('_'):
                source, target = target, source
                relation = relation[1:]
            
            edges.append((source, target, relation))

            # Add to middle nodes if not start or end
            if source not in start_nodes and source not in end_nodes:
                middle_nodes.add(source)
            if target not in start_nodes and target not in end_nodes:
                middle_nodes.add(target)

    return edges, start_nodes, middle_nodes, end_nodes

def visualize_graph(edges, start_nodes, middle_nodes, end_nodes):
    """Visualize the graph with colored nodes based on type"""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    def get_y_positions(nodes, center_y=0):
        n = len(nodes)
        if n == 0:
            return {}
        spacing = 100
        start_y = center_y - (n-1) * spacing/2
        return {node: start_y + i*spacing for i, node in enumerate(nodes)}
    
    # Get y positions for each group
    start_y_pos = get_y_positions(start_nodes)
    middle_y_pos = get_y_positions(middle_nodes)
    end_y_pos = get_y_positions(end_nodes)
    
    def add_node_with_color(node, x, y, fixed=False):
        node_type = node.split('::')[0]  # Get prefix before '::'
        color = get_node_color(node_type)
        net.add_node(
            node,
            label=node.split('::')[1],
            x=x,
            y=y,
            fixed=fixed,
            physics=not fixed,
            color=color,
            title=f"Type: {node_type}"  # Hover text
        )
    
    # Add nodes with colors
    for node in start_nodes:
        add_node_with_color(node, -400, start_y_pos[node], fixed=True)
    
    for node in middle_nodes:
        add_node_with_color(node, 0, middle_y_pos.get(node, 0))
    
    for node in end_nodes:
        add_node_with_color(node, 400, end_y_pos[node], fixed=True)
    
    # Add edges
    for source, target, relation in edges:
        net.add_edge(source, target, label=relation, arrows='to', length=200)
    
    # Configure physics and layout options
    net.set_options('''
        const options = {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -4000,
                    "centralGravity": 0.1,
                    "springLength": 200,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 1
                },
                "stabilization": {
                    "enabled": true,
                    "iterations": 2000
                }
            },
            "edges": {
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "horizontal"
                }
            },
            "layout": {
                "hierarchical": {
                    "enabled": false
                }
            }
        }
    ''')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name


def mock_predict_ddi(search_input):
    return finalize_predicion("Compound::DB01234", "CiC")

def main():
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "explanations" not in st.session_state:
        st.session_state.explanations = None

    # Data
    drugs = load_drugs()

    # Title
    st.title("DDI Prediction")

    # Form
    with st.form(key="search_form"):
        search_input = st.text_input("Source Drug:")
        submit_button = st.form_submit_button(label="Predict")

    if submit_button and search_input:
        if search_input in drugs.index:
            st.session_state.predictions, st.session_state.explanations = predict_ddi(search_input)
        else:
            st.error("DrugBank ID not found in the Knowledge Graph.")
            st.session_state.predictions = None

    if st.session_state.predictions is not None:
        # Display predictions in an interactive grid
        gb = GridOptionsBuilder.from_dataframe(st.session_state.predictions)
        gb.configure_default_column(editable=False, sortable=False)
        gb.configure_selection('single')
        grid_options = gb.build()

        response = AgGrid(
            st.session_state.predictions,
            gridOptions=grid_options,
            height=300,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED
        )

    if st.session_state.explanations is not None and response["selected_rows"] is not None:
        st.title("Explanation")
        interaction = response["selected_rows"].get("Known Interaction").iloc[0]
        if interaction is not None:
            st.write(interaction.replace("Source Drug", search_input).replace("Target Drug", response["selected_rows"].get("Target Drug").iloc[0]))
        else:
            st.write(f'The pharmacokinetics of {response["selected_rows"].get("Target Drug").iloc[0]} can be influenced when combined with {search_input}.')


        graph_data = st.session_state.explanations[st.session_state.explanations[3] == "Compound::" + response["selected_rows"].get("Target Drug").iloc[0]]
        # Process data and create graph
        edges, start_nodes, middle_nodes, end_nodes = construct_graph(graph_data)
        html_file = visualize_graph(edges, start_nodes, middle_nodes, end_nodes)

        # Display the graph in Streamlit
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=600)

        # Cleanup temporary file
        os.unlink(html_file)

if __name__ == "__main__":
    main()
