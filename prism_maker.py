import numpy as np
import networkx as nx
from landing_sim import ready_data_for_sim, descent_trajectory
from estimate_adv_prob import read_logs, AdvSuccessProb
import json
from tqdm import tqdm


def make_prism_module_file(G, satisfy, otherleaf, fpath):
    with open(fpath, 'w') as f:
        f.write("dtmc\n\n")
        f.write("module landing\n\n")
        # Do stuff with tabs

        # Number of intermediate nodes
        f.write("s: [0..%d] init 0;\n" % len(G.nodes))

        # Number of leaf nodes
        f.write("d: [0..1] init 0;\n\n")

        # Add edge data
        for node, node_data in tqdm(G.nodes(data=True)):
            edges = G.out_edges(node, data=True)

            # Nothing to do if leaf nod
            if len(edges) == 0:
                continue

            f.write("[] s=%d -> " % (node))

            line = []
            so_far = 0

            for i, (_, to, edge_data) in enumerate(edges):
                prob = edge_data["weight"]
                # Make sure probabilities sum up to 1
                if i == len(edges) - 1:
                    prob = 1 - so_far
                so_far += prob

                line.append("%f : (s'=%d)" % (prob, to))
            
            line = " + ".join(line)
            f.write(line + ";")
            f.write(" //delta: %f\n" % node_data["deviation"])

        f.write("\n")
        # Connection from non-satisfied leaf nodes to D0
        for i in otherleaf:
            f.write("[] s=%d -> (d'=0) & (s'=%d); // Safe landing leaf-node\n" % (i, len(G.nodes)))
        # Connection from satisfied nodes to D1
        # Connection from non-satisfied leaf nodes to D0
        for i in satisfy:
            f.write("[] s=%d -> (d'=1) & (s'=%d); // Unsafe landing leaf-node\n" % (i, len(G.nodes)))

        f.write("\nendmodule\n")


def get_config_class():
    with open("./config.json", 'r') as f:
        return json.load(f)


class Quantizer:
    def __init__(self, config):
        self.thresholds = []
        self.angles = []
        quant = config["angle_quantize"]
        angles_max = config["angles_max"]
        for a in angles_max:
            self.thresholds.append(a[0])
            self.angles.append(np.arange(quant, a[1]+quant, quant))
    
    def get_angles(self, z):
        for t, a in zip(self.thresholds, self.angles):
            if z <= t:
                return a
        return self.angles[-1]


def make_states(descent, config, prob_obj, quantize=5):
    H, S = ready_data_for_sim(descent)
    quantizer = Quantizer(config)
    G = nx.DiGraph()

    # Add start state
    G.add_node(0, deviation=0)
    parents, i = [0], 1

    # Finegrained data will lead to huge graph
    H = H[::quantize]
    S = S[::quantize]

    iterator = tqdm(enumerate(zip(H, S)), total=len(H))
    for t, (h, s) in iterator:
        # Compute direct distance from airport
        d_dist = (h**2 + s**2) ** 0.5
        adv_dist = config["adv_dist_multiplier"] * d_dist

        # Get probability of adversary success at this point
        adv_prob = prob_obj.prob(adv_dist)

        # Keep track of nodes made in this timestep
        # since they will be the parent nodes in next timestep
        made_in_this_iter = []

        # Iterate through all nodes at this timestep
        for parent in parents:
            parent_deviation = G.nodes[parent]['deviation']

            # Add no-deviation transition
            G.add_node(i, deviation=parent_deviation)
            made_in_this_iter.append(i)

            # Pilot takes control, adversary cannot do anything anymore
            if h <= config["decision_height"]:
                adv_prob = 0
        
            if adv_prob == 0:
                # Add no-deviation edge
                G.add_edge(parent, i, weight=1)
                i += 1
            else:
                # Add no-deviation edge
                G.add_edge(parent, i, weight=np.round(1-adv_prob, 3))
                i += 1

                # Split prob into angles based on quantization
                # Weighted based on inverse of possibilities
                angles = quantizer.get_angles(h)
                weights = 1 / angles
                weights /= np.sum(weights)
                # Round off, make sure still sums to 1
                weights *= adv_prob
                weights = np.round(weights, 3)
                
                for j in range(len(angles)):
                    # Compute deviation based on angle
                    deviation = np.tan(np.deg2rad(angles[j])) * adv_dist

                    G.add_node(i, deviation=deviation + parent_deviation)
                    made_in_this_iter.append(i)

                    # Compute transition probability
                    transition_prob = weights[j]
                    G.add_edge(parent, i, weight=transition_prob)
                    i += 1
        
        # Update set of parents
        parents = made_in_this_iter[:]
        iterator.set_description("G: %d nodes" % len(G.nodes))

    # Parent nodes at the end must be leaf nodes
    # Check which ones of them satisfy adversary's goal
    nosatisfy, satisfy = [], []
    for p in parents:
        if G.nodes[p]['deviation'] >= config["strip_width"]:
            satisfy.append(p)
        else:
            nosatisfy.append(p)

    # Return graph
    return G, satisfy, nosatisfy


if __name__ == "__main__":
    # Read adversarial data
    data = read_logs("./logs")
    prob_obj = AdvSuccessProb(data)
    
    plane = 'b789' # Boeing 787-9

    descent = descent_trajectory(plane)
    config = get_config_class()
    G, picked, otherleaf = make_states(descent, config, prob_obj)

    # Make DTCM for Prism
    make_prism_module_file(G, picked, otherleaf, "./temp.pm")
