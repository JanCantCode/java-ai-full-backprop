package tk.jandev;


import tk.jandev.neuralnetwork.NeuralNetwork;
import tk.jandev.neuralnetwork.activations.Activation;
import tk.jandev.neuralnetwork.layers.HiddenLayer;
import tk.jandev.neuralnetwork.layers.InputLayer;

import java.text.DecimalFormat;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[] trainingInput = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
        double[] trainingOutput = new double[]{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0};


        InputLayer inputLayer = new InputLayer(1);

        HiddenLayer hiddenLayer = new HiddenLayer(1, 1, Activation.IDENTITY);
        HiddenLayer hiddenLayer2 = new HiddenLayer(1, 1, Activation.IDENTITY);
        HiddenLayer hiddenLayer3 = new HiddenLayer(1, 1, Activation.IDENTITY);
        HiddenLayer hiddenLayer4 = new HiddenLayer(1, 1, Activation.IDENTITY);
        HiddenLayer outputLayer = new HiddenLayer(1, 1, Activation.IDENTITY);

        NeuralNetwork network = new NeuralNetwork(inputLayer, new HiddenLayer[]{hiddenLayer, hiddenLayer2, hiddenLayer3, hiddenLayer4, outputLayer});

        System.out.println("output of 1.0 "+Arrays.toString(network.feed(new double[]{100.0})));
        System.out.println("training lol");

        for (int i = 0; i < trainingInput.length; i++) {
            network.fullBackpropagation(new double[]{trainingInput[i]}, new double[]{trainingOutput[i]}, 0.001);
        }

        System.out.println("After training: "+Arrays.toString(network.feed(new double[]{100.0})));

    }

    public static double[][] toDarray(double[] input) {
        double[][] output = new double[input.length][];

        for (int i = 0; i < input.length; i++) {
            output[i] = new double[]{input[i]};
        }
        return output;
    }
}