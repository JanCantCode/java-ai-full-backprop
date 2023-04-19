package tk.jandev.neuralnetwork.layers;

public class InputLayer extends Layer {
    private final int nodesOut;

    public InputLayer(int nodesOut) {
        super(nodesOut);
        this.nodesOut = nodesOut;
    }

    @Override
    public double[] forward(double[] inputs) {
        if (inputs.length != this.nodesOut) {
            System.out.println("Nodes in input layer were not equal to input! Nodes: " +this.nodesOut+" Inputs: " + inputs.length);
        }
        return inputs;
    }
}
