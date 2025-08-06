# model_3d.py
import graphviz

def plot_3d_model_graph(model, title="Model Architecture 3D"):
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='TB', size='8,10')
    dot.attr('node', shape='box', style='rounded', fontsize='12')

    prev = "Input"
    dot.node(prev, "Input", color="lightblue2", style="filled")

    idx = 0
    for layer in model:
        if isinstance(layer, nn.Flatten):
            continue
        name = f"Layer{idx}_{layer.__class__.__name__}"
        label = layer.__class__.__name__
        if hasattr(layer, 'out_channels'):
            label += f"({layer.out_channels})"
        elif hasattr(layer, 'out_features'):
            label += f"({layer.out_features})"
        dot.node(name, label, color="lightgrey" if "Conv" in label else "wheat")
        dot.edge(prev, name)
        prev = name
        idx += 1

    dot.node("Output", "Output", color="lightblue2", style="filled")
    dot.edge(prev, "Output")
    return dot