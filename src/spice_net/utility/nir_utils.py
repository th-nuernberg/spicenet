
from spice_net.learning_rate_functions.learning_rate_function import LearningRateFunction
from ..spice_net_som import SpiceNetSom
from ..spice_net_hcm import SpiceNetHcm
from ..spice_net import SpiceNet
import nir
import numpy as np

def nir_to_spicenet(
        nir_spicenet: nir.SPICENet,
        som_lrf_tuning_curve: LearningRateFunction,
        som_lrf_interaction_kernel: LearningRateFunction,
        hcm_lrf_weights: LearningRateFunction,
        hcm_trust_of_new: LearningRateFunction
    ) -> SpiceNet | dict[str, SpiceNet]:
    """Converts a nir spicenet to one or multiple spicenets depending on the number of SOMs."""    
    # Transform all soms to spicenet soms
    spicenet_soms = {som_name: nir_to_som(nir_som, som_lrf_tuning_curve, som_lrf_interaction_kernel) for som_name, nir_som in nir_spicenet.soms.items()}
    spicenet_values_list = list(spicenet_soms.values())
    spicenet_keys_list = list(spicenet_soms.keys())
    if len(spicenet_values_list) == 2:
        # If there are only two SOMs, create a single SpiceNet
        return SpiceNet(correlation_matrix=SpiceNetHcm(spicenet_values_list[0], spicenet_values_list[1], hcm_lrf_weights, hcm_trust_of_new))
    
    # Otherwise we need to create multiple SpiceNets - one for each combination
    spicenet_combos = []
    for key in range(0, len(spicenet_keys_list)):
        for key2 in range(key + 1, len(spicenet_keys_list)):
            spicenet_combos.append((key, key2))
            
    spicenets = {}
    for combo in spicenet_combos:
        spicenets[f"{combo[0]}_{combo[1]}"] = SpiceNet(correlation_matrix=SpiceNetHcm(spicenet_soms[combo[0]], spicenet_soms[combo[1]], hcm_lrf_weights, hcm_trust_of_new))
        
    return spicenets

def spicenet_to_nir(spicenet: SpiceNet) -> nir.SPICENet:
    """Converts a single SPICEnet to a NIR SPICEnet"""
    nir_spicenet = nir.SPICENet.from_list([som_to_nir(spicenet.som_1), som_to_nir(spicenet.som_2)])
    return nir_spicenet

def som_to_nir(SpiceNetSom: SpiceNetSom) -> nir.SPICEnetSOM:
    nir_som_neurons = [som_neuron_to_nir(neuron) for neuron in SpiceNetSom.neurons]
    return nir.SPICEnetSOM(nir_som_neurons)

def nir_to_som(nir_spicenet_som: nir.SPICEnetSOM, lrf_tuning_curve: LearningRateFunction, lrf_interaction_kernel: LearningRateFunction) -> SpiceNetSom:
    """Converts NIR SPICEnetSOM to SpiceNetSom.
    
    Requires the learning rate functions for the tuning curve and interaction kernel as NIR does not transfer this.

    Args:
        nir_spicenet_som (nir.SPICEnetSOM): NIR SPICEnetSOM to convert to SpiceNetSom
        lrf_tuning_curve (LearningRateFunction): Tuning curve learning rate function
        lrf_interaction_kernel (LearningRateFunction): Interaction kernel learning rate function

    Returns:
        SpiceNetSom: SPICE Net SOM
    """
    # Initialize SpiceNetSom with correct number of SOM neurons
    # min and max value are not important as we will override all neurons shortly
    spicenet_som = SpiceNetSom(len(nir_spicenet_som.neurons), 0, 0, lrf_tuning_curve, lrf_interaction_kernel)
    
    # Override all neurons with the ones from the NIR SPICEnetSOM
    for i, nir_neuron in enumerate(nir_spicenet_som.neurons):
        spicenet_som.neurons[i] = nir_to_som_neuron(nir_neuron)
        
    return spicenet_som

def som_neuron_to_nir(spicenet_som_neuron: SpiceNetSom.SomNeuron) -> nir.SPICEnetSOMNeuron:
    return nir.SPICEnetSOMNeuron(np.array(spicenet_som_neuron.tuning_curve_width), np.array(spicenet_som_neuron.preferred_value))

def nir_to_som_neuron(nir_spicenet_som_neuron: nir.SPICEnetSOMNeuron) -> SpiceNetSom.SomNeuron:
    return SpiceNetSom.SomNeuron(nir_spicenet_som_neuron.mean.item(), nir_spicenet_som_neuron.std.item())