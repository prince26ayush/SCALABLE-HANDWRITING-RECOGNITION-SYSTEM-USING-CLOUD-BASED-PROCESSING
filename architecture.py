from graphviz import Digraph

def create_architecture_diagram(output_file="architecture_diagram"):
    """
    Creates a well-labeled architecture diagram using Graphviz.

    Args:
        output_file (str): The name of the output file (without extension).
    """

    dot = Digraph(comment='Scalable Handwriting Recognition System Architecture')

    # Nodes
    dot.node('S3', 'Handwritten Document Images (S3)', shape='cylinder')
    dot.node('EMR', 'EMR Cluster', shape='box')
    dot.node('Spark', 'Spark Processing', shape='ellipse')
    dot.node('OpenCV', 'Image Preprocessing (OpenCV)', shape='box')
    dot.node('HOG', 'Feature Extraction (HOG)', shape='box')
    dot.node('ML', 'Machine Learning Model (TensorFlow/PyTorch)', shape='box')
    dot.node('Results', 'Handwriting Recognition Results (CSV/Database)', shape='cylinder')
    dot.node('HDFS', 'HDFS', shape='cylinder')
    dot.node('User', 'User/Application', shape='house')
    dot.node('CloudA', 'Cloud Provider A', shape='cloud')
    dot.node('CloudB', 'Cloud Provider B', shape='cloud')
    dot.node('CloudC', 'Cloud Provider C', shape='cloud')

    # Edges (Connections)
    dot.edge('S3', 'EMR', label='Data Input')
    dot.edge('EMR', 'Spark', label='Processing')
    dot.edge('Spark', 'OpenCV', label='Image Data')
    dot.edge('OpenCV', 'HOG', label='Processed Images')
    dot.edge('HOG', 'ML', label='Extracted Features')
    dot.edge('ML', 'Results', label='Recognition Output')
    dot.edge('EMR', 'HDFS', label='Storage')
    dot.edge('Spark', 'HDFS', label='Intermediate Data')
    dot.edge('User', 'Results', label='Data Access')
    dot.edge('CloudA', 'EMR', label='Cloud Deployment')
    dot.edge('CloudB', 'EMR', label='Cloud Deployment')
    dot.edge('CloudC', 'EMR', label='Cloud Deployment')

    # Subgraphs
    with dot.subgraph(name='cluster_Cloud') as c:
        c.attr(label="Cloud Environment")
        c.node('EMR')
        c.node('HDFS')

    with dot.subgraph(name='cluster_CrossCloud') as cc:
        cc.attr(label="Cross Cloud Implementation")
        cc.node('CloudA')
        cc.node('CloudB')
        cc.node('CloudC')

    # Render the graph to a file (e.g., PNG, PDF)
    dot.render(output_file, format='png', cleanup=True)
    print(f"Architecture diagram saved to {output_file}.png")

if __name__ == "__main__":
    create_architecture_diagram()