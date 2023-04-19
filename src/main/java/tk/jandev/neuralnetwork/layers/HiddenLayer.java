package tk.jandev.neuralnetwork.layers;

import tk.jandev.neuralnetwork.Node;
import tk.jandev.neuralnetwork.activations.Activation;
import tk.jandev.neuralnetwork.utils.Helper;

public class HiddenLayer extends Layer {
    private final Node[] nodes;
    public HiddenLayer(int nodesIn, int nodesOut, Activation activation) {
        super(nodesIn, nodesOut);
        this.nodes = Helper.makeNodes(nodesOut, activation, nodesIn);
    }

    @Override
    public double[] forward(double[] inputs) {
        for (Node node : this.nodes) {
            node.feed(inputs);
        }

        return this.getActivations();
    }

    public double[] getActivations() {
        double[] activations = new double[this.nodes.length];

        for (int i = 0; i < this.nodes.length; i++) {
            activations[i] = this.nodes[i].getActivation();
        }

        return activations;
    }

    public Node[] getNodes() {
        return this.nodes;
    }


}
