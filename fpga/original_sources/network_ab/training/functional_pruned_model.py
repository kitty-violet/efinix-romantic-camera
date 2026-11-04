"""Physical deletion of experimentally dispensable FX branches."""
import torch
from .importance_pruned_teacher import build_group_pruned


def remove_attention(net):
    graph=net.generator.graph
    target='generator_1/main/External_attention/add_1'
    nodes=[n for n in graph.nodes if n.op=='call_module' and n.target==target];assert len(nodes)==1
    node=nodes[0];assert node.args[1].target=='generator_1/base/LeakyRelu_6'
    # Keep the following activation. This is exactly the attention alpha=0
    # intervention, including the author's second leaky-ReLU.
    node.replace_all_uses_with(node.args[1]);graph.erase_node(node)
    graph.eliminate_dead_code();graph.lint();net.generator.recompile();net.generator.delete_all_unused_submodules()
    assert not any('External_attention' in n.target and n.target!='generator_1/main/External_attention/LeakyRelu' for n in graph.nodes if n.op=='call_module')
    assert sum(isinstance(m,torch.nn.Conv2d) for m in net.modules())==27
    return net


def build(root,state):
    net=build_group_pruned(root,state['indices'])
    if state.get('removed_attention'):remove_attention(net)
    net.load_state_dict(state['generator']);return net
