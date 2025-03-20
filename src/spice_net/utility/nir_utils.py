
from spice_net.learning_rate_functions import *
from ..spice_net_som import SpiceNetSom
from ..spice_net_hcm import SpiceNetHcm
from ..spice_net import SpiceNet
import nir
import numpy as np
from typing import Union

SpiceTypes = Union[SpiceNet, SpiceNetHcm, SpiceNetSom, SpiceNetSom.SomNeuron]

def from_nir(node: nir.NIRNode,
            som_1: SpiceNetSom = None,
            som_2: SpiceNetSom = None,
            som_lrf_tuning_curve: LearningRateFunction | None = None,
            som_lrf_interaction_kernel: LearningRateFunction | None = None,
            hcm_lrf_weights: LearningRateFunction | None = None,
            hcm_trust_of_new: LearningRateFunction | None = None) -> SpiceTypes | list[SpiceTypes]:
    """This function converts NIR nodes back to their SPICEnet representation. This can either be a single SPICEnet component or a whole list of components mapped in a NIR Graph.

    Args:
        node (nir.NIRNode): NIR node to convert to SPICEnet object
        som_1 (SpiceNetSom, optional): Only needed if trying to import single HCMs. Defaults to None.
        som_2 (SpiceNetSom, optional): Only needed if trying to import single HCMs. Defaults to None.
        som_lrf_tuning_curve (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        som_lrf_interaction_kernel (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_lrf_weights (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_trust_of_new (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.

    Returns:
        SpiceTypes | list[SpiceTypes]: Returns a single SPICEnet component or a list of components of the original SPICEnet implementation.
    """
    # If the node is a graph, we need to iterate over all nodes
    return _from_nir(node, False, som_1, som_2, som_lrf_tuning_curve, som_lrf_interaction_kernel, hcm_lrf_weights, hcm_trust_of_new)
        
def _from_nir(node: nir.NIRNode, 
            recurse: bool = False,
            som_1: SpiceNetSom = None,
            som_2: SpiceNetSom = None,
            som_lrf_tuning_curve: LearningRateFunction | None = None,
            som_lrf_interaction_kernel: LearningRateFunction | None = None,
            hcm_lrf_weights: LearningRateFunction | None = None,
            hcm_trust_of_new: LearningRateFunction | None = None) -> SpiceTypes | list[SpiceTypes]:
    # If the node is a graph, we need to iterate over all nodes
    if isinstance(node, nir.NIRGraph):
        result = []
        for _, subnode in node.nodes.items():
            result.extend(_from_nir(subnode, True, som_1, som_2, som_lrf_tuning_curve, som_lrf_interaction_kernel, hcm_lrf_weights, hcm_trust_of_new))
    else:
        result = [_map_from_nir(node, som_1, som_2, som_lrf_tuning_curve, som_lrf_interaction_kernel, hcm_lrf_weights, hcm_trust_of_new)]
    
    result = [x for x in result if x is not None]
    if len(result) == 1 and not recurse:
        return result[0]
    return result
    
def _map_from_nir(node: nir.NIRNode,
                som_1: SpiceNetSom = None,
                som_2: SpiceNetSom = None,
                som_lrf_tuning_curve: LearningRateFunction | None = None,
                som_lrf_interaction_kernel: LearningRateFunction | None = None,
                hcm_lrf_weights: LearningRateFunction | None = None,
                hcm_trust_of_new: LearningRateFunction | None = None) -> Union[SpiceNet, SpiceNetHcm, SpiceNetSom, SpiceNetSom.SomNeuron]:
    if isinstance(node, nir.SPICENet):
        return nir_to_spicenet(node, som_lrf_tuning_curve, som_lrf_interaction_kernel, hcm_lrf_weights, hcm_trust_of_new)
    elif isinstance(node, nir.SPICEnetHCM):
        return nir_to_hcm(node, som_1, som_2, hcm_lrf_weights, hcm_trust_of_new)
    elif isinstance(node, nir.SPICEnetSOM):
        return nir_to_som(node, som_lrf_tuning_curve, som_lrf_interaction_kernel)
    elif isinstance(node, nir.SPICEnetSOMNeuron):
        return nir_to_som_neuron(node)
    elif isinstance(node, nir.Input):
        None
    elif isinstance(node, nir.Output):
        None
    else:
        raise TypeError(f"Unsupported node type {type(node)}")

def to_nir(nodes: SpiceTypes | list[SpiceTypes]) -> nir.NIRNode:
    """Transforms any SPICEnet component to a NIR node. Can also transform multiple components at once by passing a list of components.

    Args:
        nodes (SpiceTypes | list[SpiceTypes]): Component or list of components to transform to NIR

    Returns:
        nir.NIRNode: NIR graph with the components
    """
    if isinstance(nodes, list):
        return nir.NIRGraph.from_list([_map_to_nir(node) for node in nodes])
    return nir.NIRGraph.from_list([_map_to_nir(nodes)])

def _map_to_nir(node: SpiceTypes) -> nir.NIRNode:
    if isinstance(node, SpiceNet):
        return spicenet_to_nir(node)
    elif isinstance(node, SpiceNetHcm):
        return hcm_to_nir(node)
    elif isinstance(node, SpiceNetSom):
        return som_to_nir(node)
    elif isinstance(node, SpiceNetSom.SomNeuron):
        return som_neuron_to_nir(node)
    else:
        raise TypeError(f"Unsupported node type {type(node)}")

def nir_to_spicenet(
        nir_spicenet: nir.SPICENet,
        som_lrf_tuning_curve: LearningRateFunction | None = None,
        som_lrf_interaction_kernel: LearningRateFunction | None = None,
        hcm_lrf_weights: LearningRateFunction | None = None,
        hcm_trust_of_new: LearningRateFunction | None = None
    ) -> SpiceNet | dict[str, SpiceNet]:
    """Converts a nir spicenet to one or multiple spicenets depending on the number of SOMs.

    Args:
        nir_spicenet (nir.SPICENet): object to import
        som_lrf_tuning_curve (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        som_lrf_interaction_kernel (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_lrf_weights (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_trust_of_new (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.

    Returns:
        SpiceNet | dict[str, SpiceNet]: Returns the SPICEnet object or a dictionary of SPICEnet objects if multiple SOMs and HCMs were present in the NIR SPICEnet.
    """
    
    # Transform all soms to spicenet soms
    spicenet_soms = {som_name: nir_to_som(nir_som, som_lrf_tuning_curve, som_lrf_interaction_kernel) for som_name, nir_som in nir_spicenet.soms.items()}
    spicenet_values_list = list(spicenet_soms.values())
    spicenet_keys_list = list(spicenet_soms.keys())
    
    # Otherwise we need to create multiple SpiceNets - one for each combination
    spicenet_combos = []
    for key in range(0, len(spicenet_keys_list)):
        for key2 in range(key + 1, len(spicenet_keys_list)):
            spicenet_combos.append((key, key2))
            
    spicenets = {}
    for combo in spicenet_combos:
        if nir_spicenet.hcms.get(f"{combo[0]}_{combo[1]}"):
            spicenets[f"{combo[0]}_{combo[1]}"] = SpiceNet(correlation_matrix=nir_to_hcm(nir_spicenet.hcms[f"{combo[0]}_{combo[1]}"], spicenet_soms[f"{combo[0]}"], spicenet_soms[f"{combo[1]}"], hcm_lrf_weights, hcm_trust_of_new))
        else:
            spicenets[f"{combo[1]}_{combo[0]}"] = SpiceNet(correlation_matrix=nir_to_hcm(nir_spicenet.hcms[f"{combo[1]}_{combo[0]}"], spicenet_soms[f"{combo[1]}"], spicenet_soms[f"{combo[0]}"], hcm_lrf_weights, hcm_trust_of_new))
        
    if len(list(spicenets.keys())) == 1:
        return list(spicenets.values())[0]
        
    return spicenets

def spicenet_to_nir(spicenet: SpiceNet) -> nir.SPICENet:
    """Converts a single SPICEnet to a NIR SPICEnet

    Args:
        spicenet (SpiceNet): SPICEnet to convert

    Returns:
        nir.SPICENet: NIR SPICEnet object
    """
    nir_spicenet = nir.SPICENet.from_lists([som_to_nir(spicenet.som_1), som_to_nir(spicenet.som_2)], [(0, 1, hcm_to_nir(spicenet.get_correlation_matrix()))])
    return nir_spicenet

def nir_to_hcm(nir_spicenet_hcm: nir.SPICEnetHCM, som_1: SpiceNetSom, som_2: SpiceNetSom, hcm_lrf_weights: LearningRateFunction | None = None, hcm_trust_of_new: LearningRateFunction | None = None) -> SpiceNetHcm:
    """Convert NIR HCM to original SPICEnet HCM.

    Args:
        nir_spicenet_hcm (nir.SPICEnetHCM): object to convert
        som_1 (SpiceNetSom): SOM 1 of the HCM.
        som_2 (SpiceNetSom): SOM 2 of the HCM.
        hcm_lrf_weights (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_trust_of_new (LearningRateFunction | None, optional): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.

    Returns:
        SpiceNetHcm: Original SPICEnet HCM object
    """
    if hcm_lrf_weights is None:
        if "lrf_weights" in nir_spicenet_hcm.metadata:
            hcm_lrf_weights = get_lr_function_from_metadata(nir_spicenet_hcm.metadata["lrf_weights"])
    if hcm_trust_of_new is None:
        if "lrf_trust_of_new" in nir_spicenet_hcm.metadata:
            hcm_trust_of_new = get_lr_function_from_metadata(nir_spicenet_hcm.metadata["lrf_trust_of_new"])
    
    hcm = SpiceNetHcm(som_1=som_1, som_2=som_2, lrf_weights=hcm_lrf_weights, lrf_trust_of_new=hcm_trust_of_new)
    
    if "iteration" in nir_spicenet_hcm.metadata:
        hcm.set_iteration(nir_spicenet_hcm.metadata["iteration"])
    
    # Override weights with the ones from the NIR SPICEnetHCM
    # We need to use copy to not pass a reference
    hcm.weights = nir_spicenet_hcm.weights.copy()
    hcm.activation_bar_vector_1 = nir_spicenet_hcm.activation_bar_vector_1.copy()
    hcm.activation_bar_vector_2 = nir_spicenet_hcm.activation_bar_vector_2.copy()
    
    return hcm

def hcm_to_nir(spicenet_hcm: SpiceNetHcm) -> nir.SPICEnetHCM:
    """Transforms a SPICEnet HCM to a NIR SPICEnet HCM.

    Args:
        spicenet_hcm (SpiceNetHcm): object to transform

    Returns:
        nir.SPICEnetHCM: NIR SPICEnet HCM object
    """
    # We need to use a deep copy here. This is so we avoid passing a reference through NIR and then possibly import a reference in another python object in another framework.
    nir_hcm = nir.SPICEnetHCM(weights=spicenet_hcm.weights.copy(),
                              activation_bar_vector_1=spicenet_hcm.activation_bar_vector_1.copy(),
                              activation_bar_vector_2=spicenet_hcm.activation_bar_vector_2.copy(),
                              metadata={
                                "lrf_trust_of_new": get_metadata_for_lr_function(spicenet_hcm.get_trust_of_new_lrf()),
                                "lrf_weights": get_metadata_for_lr_function(spicenet_hcm.get_weights_lrf()),
                                "iteration": spicenet_hcm.get_iteration()
                              })
    return nir_hcm

def som_to_nir(som: SpiceNetSom) -> nir.SPICEnetSOM:
    """Transforms a SPICEnet SOM to a NIR SPICEnet SOM.

    Args:
        som (SpiceNetSom): object to transform

    Returns:
        nir.SPICEnetSOM: NIR SPICEnet SOM object
    """
    nir_som_neurons = [som_neuron_to_nir(neuron) for neuron in som.neurons]
    return nir.SPICEnetSOM(nir_som_neurons, 
                           metadata={
                               "lrf_tuning_curve": get_metadata_for_lr_function(som.get_lrf_tuning_curve()),
                               "lrf_interaction_kernel": get_metadata_for_lr_function(som.get_lrf_interaction_kernel()),
                               "iteration": som.get_iteration()
                              }
                           )

def nir_to_som(nir_spicenet_som: nir.SPICEnetSOM, lrf_tuning_curve: LearningRateFunction | None = None, lrf_interaction_kernel: LearningRateFunction | None = None) -> SpiceNetSom:
    """Converts NIR SPICEnetSOM to SpiceNetSom.
    
    Requires the learning rate functions for the tuning curve and interaction kernel as NIR does not transfer this.

    Args:
        nir_spicenet_som (nir.SPICEnetSOM): NIR SPICEnetSOM to convert to SpiceNetSom
        lrf_tuning_curve (LearningRateFunction | None): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        lrf_interaction_kernel (LearningRateFunction | None): Overwrite learning rate function. Will try to infer if not provided but fail if inference not possible. Defaults to None.

    Returns:
        SpiceNetSom: SPICE Net SOM
    """
    
    if lrf_tuning_curve is None:
        if "lrf_tuning_curve" in nir_spicenet_som.metadata:
            lrf_tuning_curve = get_lr_function_from_metadata(nir_spicenet_som.metadata["lrf_tuning_curve"])
    if lrf_interaction_kernel is None:
        if "lrf_interaction_kernel" in nir_spicenet_som.metadata:
            lrf_interaction_kernel = get_lr_function_from_metadata(nir_spicenet_som.metadata["lrf_interaction_kernel"])
    
    # Initialize SpiceNetSom with correct number of SOM neurons
    # min and max value are not important as we will override all neurons shortly
    spicenet_som = SpiceNetSom(len(nir_spicenet_som.neurons), 0, 0, lrf_tuning_curve, lrf_interaction_kernel)
    
    # Override all neurons with the ones from the NIR SPICEnetSOM
    for i, nir_neuron in enumerate(nir_spicenet_som.neurons):
        spicenet_som.neurons[i] = nir_to_som_neuron(nir_neuron)
        
    if "iteration" in nir_spicenet_som.metadata:
        spicenet_som.set_iteration(nir_spicenet_som.metadata["iteration"])
        
    return spicenet_som

def som_neuron_to_nir(spicenet_som_neuron: SpiceNetSom.SomNeuron) -> nir.SPICEnetSOMNeuron:
    """Transform single SOM neuron to NIR.

    Args:
        spicenet_som_neuron (SpiceNetSom.SomNeuron): object to transform

    Returns:
        nir.SPICEnetSOMNeuron: NIR SOM neuron object
    """
    # creating new reference to avoid changing the original object
    return nir.SPICEnetSOMNeuron(np.array(spicenet_som_neuron.tuning_curve_width), np.array(spicenet_som_neuron.preferred_value))

def nir_to_som_neuron(nir_spicenet_som_neuron: nir.SPICEnetSOMNeuron) -> SpiceNetSom.SomNeuron:
    """Transform single NIR SOM neuron to SPICEnet SOM neuron.

    Args:
        nir_spicenet_som_neuron (nir.SPICEnetSOMNeuron): object to transform

    Returns:
        SpiceNetSom.SomNeuron: original SPICEnet SOM neuron object
    """
    return SpiceNetSom.SomNeuron(nir_spicenet_som_neuron.mean.item(), nir_spicenet_som_neuron.std.item())

def export_to_hdf5(filepath: str , nodes: SpiceTypes | list[SpiceTypes]):
    """Exports SPICEnet compontent(s) to a HDF5 file.

    Args:
        filepath (str): Path for file to export to
        nodes (SpiceTypes | list[SpiceTypes]): Object or list of objects to export
    """
    
    nir_graph = to_nir(nodes)
    nir.write(filepath, nir_graph)
    
def read_from_hdf5(
        filepath: str,
        som_1: SpiceNetSom = None,
        som_2: SpiceNetSom = None,
        som_lrf_tuning_curve: LearningRateFunction = None,
        som_lrf_interaction_kernel: LearningRateFunction = None,
        hcm_lrf_weights: LearningRateFunction = None,
        hcm_trust_of_new: LearningRateFunction = None
    ) -> SpiceTypes | list[SpiceTypes]:
    """This function converts NIR nodes in HDF6 format back to their SPICEnet representation. This can either be a single SPICEnet component or a whole list of components mapped in a NIR Graph.

    Args:
        filepath (str): HDF5 path to read from
        som_1 (SpiceNetSom, optional): Only needed if trying to import single HCMs. Defaults to None.
        som_2 (SpiceNetSom, optional): Only needed if trying to import single HCMs. Defaults to None.
        som_lrf_tuning_curve (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        som_lrf_interaction_kernel (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_lrf_weights (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.
        hcm_trust_of_new (LearningRateFunction | None, optional): Overwrite learning rate function on import. Will try to infer if not provided but fail if inference not possible. Defaults to None.

    Returns:
        SpiceTypes | list[SpiceTypes]: Returns a single SPICEnet component or a list of components of the original SPICEnet implementation.
    """
    
    nir_spicenet = nir.read(filepath)
    return from_nir(nir_spicenet, som_1, som_2, som_lrf_tuning_curve, som_lrf_interaction_kernel, hcm_lrf_weights, hcm_trust_of_new)

def map_lr_name_to_type(name: str) -> str:
    if name == "ConstLRF":
        return "const"
    elif name == "ExpEDecayLRF":
        return "exp"
    elif name == "InverseTimeAdaptation":
        return "inv_time"
    elif name == "LinearLRF":
        return "linear"
    else:
        raise ValueError(f"Unknown learning rate function type {name}")
    

def get_metadata_for_lr_function(lrf: LearningRateFunction) -> dict:
    return {
        "type": map_lr_name_to_type(lrf.__class__.__name__),
        "parameters": lrf.get_parameters()
    }
    
def get_lr_function_from_metadata(metadata: dict) -> LearningRateFunction | None:
    if metadata["type"] == "const":
        return ConstLRF(metadata["parameters"]["value"])
    elif metadata["type"] == "exp":
        return ExpEDecayLRF(metadata["parameters"]["speed"], metadata["parameters"]["approached_value"], metadata["parameters"]["x_shift"])
    elif metadata["type"] == "inv_time":
        return InverseTimeAdaptation.from_parameters(metadata["parameters"]["A"], metadata["parameters"]["B"], metadata["parameters"]["planned_iterations"])
    elif metadata["type"] == "linear":
        return LinearLRF(metadata["parameters"]["slope"], metadata["parameters"]["bias"])
    else:
        return None