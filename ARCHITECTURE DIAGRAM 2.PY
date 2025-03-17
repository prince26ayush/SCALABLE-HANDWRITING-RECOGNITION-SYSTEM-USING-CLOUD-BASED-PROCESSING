from graphviz import Digraph

def create_detailed_architecture_diagram(output_file="detailed_architecture_diagram"):
    """
    Creates a detailed and well-labeled architecture diagram using Graphviz.

    Args:
        output_file (str): The name of the output file (without extension).
    """

    dot = Digraph(comment='Detailed Scalable Handwriting Recognition System Architecture')

    # Nodes
    dot.node('S3', 'Handwritten Document Images (S3)', shape='cylinder')
    dot.node('EMR', 'EMR Cluster (YARN)', shape='box')
    dot.node('SparkDriver', 'Spark Driver Program', shape='ellipse')
    dot.node('SparkExecutors', 'Spark Executors', shape='box')
    dot.node('HDFS', 'HDFS (EMR Storage)', shape='cylinder')
    dot.node('ImageLoader', 'Image Data Loader (PySpark)', shape='box')
    dot.node('OpenCV', 'Image Preprocessing (OpenCV)', shape='box')
    dot.node('HOG', 'HOG Feature Extraction', shape='box')
    dot.node('ModelLoader', 'Model Loader (TensorFlow/PyTorch)', shape='box')
    dot.node('ModelInference', 'Model Inference (TensorFlow/PyTorch)', shape='box')
    dot.node('ResultWriter', 'Result Writer (CSV/Database)', shape='box')
    dot.node('User', 'User/Application', shape='house')
    dot.node('CloudA', 'Cloud Provider A', shape='cloud')
    dot.node('CloudB', 'Cloud Provider B', shape='cloud')
    dot.node('CloudC', 'Cloud Provider C', shape='cloud')
    dot.node('Logging', 'Logging (CloudWatch/Spark Logs)', shape='note')
    dot.node('Monitoring', 'Monitoring (Spark UI/YARN UI)', shape='note')

    # Edges (Connections)
    dot.edge('S3', 'EMR', label='Data Input')
    dot.edge('EMR', 'SparkDriver', label='Job Submission')
    dot.edge('SparkDriver', 'SparkExecutors', label='Tasks Distribution')
    dot.edge('SparkExecutors', 'HDFS', label='Data I/O')
    dot.edge('SparkExecutors', 'ImageLoader', label='Image Loading')
    dot.edge('ImageLoader', 'OpenCV', label='Image Data')
    dot.edge('OpenCV', 'HOG', label='Processed Images')
    dot.edge('HOG', 'ModelLoader', label='Extracted Features')
    dot.edge('ModelLoader', 'ModelInference', label='Model and Features')
    dot.edge('ModelInference', 'ResultWriter', label='Recognition Results')
    dot.edge('ResultWriter', 'HDFS', label='Result Storage')
    dot.edge('User', 'ResultWriter', label='Data Access')
    dot.edge('CloudA', 'EMR', label='Cloud Deployment')
    dot.edge('CloudB', 'EMR', label='Cloud Deployment')
    dot.edge('CloudC', 'EMR', label='Cloud Deployment')
    dot.edge('EMR', 'Logging', label='Logs')
    dot.edge('EMR', 'Monitoring', label='Metrics')

    # Subgraphs
    with dot.subgraph(name='cluster_Cloud') as c:
        c.attr(label="Cloud Environment")
        c.node('EMR')
        c.node('HDFS')
        c.node('Logging')
        c.node('Monitoring')

    with dot.subgraph(name='cluster_Spark') as s:
        s.attr(label="Spark Processing")
        s.node('SparkDriver')
        s.node('SparkExecutors')
        s.node('ImageLoader')
        s.node('OpenCV')
        s.node('HOG')
        s.node('ModelLoader')
        s.node('ModelInference')
        s.node('ResultWriter')

    with dot.subgraph(name='cluster_CrossCloud') as cc:
        cc.attr(label="Cross Cloud Implementation")
        cc.node('CloudA')
        cc.node('CloudB')
        cc.node('CloudC')

    # Render the graph to a file (e.g., PNG, PDF)
    dot.render(output_file, format='png', cleanup=True)
    print(f"Detailed architecture diagram saved to {output_file}.png")

if __name__ == "__main__":
    create_detailed_architecture_diagram()