package tk.jandev.neuralnetwork.utils;

import tk.jandev.neuralnetwork.Node;
import tk.jandev.neuralnetwork.activations.Activation;

public class Helper {
    public static Node[] makeNodes(int amount, Activation algorithm, int amountInPrevious) {
        Node[] nodes = new Node[amount];

        for (int i = 0; i < amount; i++) {
            nodes[i] = new Node(algorithm);
            nodes[i].setWeights(makeNumbers(amountInPrevious));
        }

        return nodes;
    }

    public static double[] makeNumbers(int amount) {
        double[] numbers = new double[amount];

        for (int i = 0; i < amount; i++) {
            numbers[i] = Math.random() + 0.5;
        }

        return numbers;
    }
}
