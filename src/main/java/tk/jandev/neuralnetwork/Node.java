package tk.jandev.neuralnetwork;

import tk.jandev.neuralnetwork.activations.Activation;

import java.util.Arrays;

public class Node {
    private double acitvation;
    private double temporary;
    private double[] weights;
    public double delta;
    private final Activation algorithm;

    public Node(Activation algorithm) {
        this.algorithm = algorithm;
    }




    public double feed(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        this.acitvation = this.algorithm.calculate(sum);
        return this.acitvation;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getActivation() {
        return this.acitvation;
    }

    public double[] getWeights() {
        return this.weights;
    }

    public void setWeight(int index, double weight) {
        this.weights[index] = weight;
    }



}
