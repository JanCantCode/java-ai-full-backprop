package tk.jandev.neuralnetwork.layers;

public abstract class Layer {
    public Layer(int nodesIn, int nodesOut) {

    }

    public Layer(int nodesOut) {

    }

    public abstract double[] forward(double[] inputs);


}
