package tk.jandev.neuralnetwork;

import tk.jandev.neuralnetwork.layers.HiddenLayer;
import tk.jandev.neuralnetwork.layers.InputLayer;
import tk.jandev.neuralnetwork.layers.Layer;

public class NeuralNetwork {
    private final InputLayer inputLayer;
    private final HiddenLayer[] layers;
    public NeuralNetwork(InputLayer inputLayer, HiddenLayer[] layers) {
        this.layers = layers;
        this.inputLayer = inputLayer;
    }

    public double[] feed(double[] input) {
        input = this.inputLayer.forward(input); // why am I doing this? I don't know, it just returns itsself!

        for (Layer layer : this.layers) {
            input = layer.forward(input);
        }

        return input;
    }

    public void trainSingleLayerPerceptron(double[] trainingInput, double[] expectedOutput, double learningRate) {
        if (this.layers.length != 1) {
            System.out.println(this.layers.length);
            System.out.println("Network is beeing backwards propagated with non output layers, but only uses hidden layer propagation!!!");
        }




        this.feed(trainingInput);

        HiddenLayer outputLayer = this.layers[this.layers.length-1]; // We can only propagate in a hidden / output layer

        for (Node node : outputLayer.getNodes()) {
            for (int i = 0; i < node.getWeights().length; i++) {
                double error = expectedOutput[i] - node.getActivation();
                double deltaWeight = error * trainingInput[i] * learningRate;
                node.setWeight(i, node.getWeights()[i] + deltaWeight);
            }
        }


    }

    public double[] calculateError(double[] input, double[] expectedOutput) {
        double[] actualOutput = this.feed(input);
        double[] errors = new double[input.length];

        for (int i = 0; i < actualOutput.length; i++) {
            errors[i] = this.cost(actualOutput[i], expectedOutput[i]);
        }

        return errors;
    }

    public double cost(double output, double expected) {
        return Math.pow((expected - output), 2);
    }

    public double networkCost(double[] input, double[] expected) {
        double[] output = this.feed(input);
        double cost = 0;

        for (int i = 0; i < expected.length; i++) {
            cost += cost(output[i], expected[i]);
        }

        return cost / expected.length;
    }

    public double costOverall(double[][] inputs, double[][] expected) {
        double cost = 0;
        for (int i = 0; i < inputs.length; i++) {
            cost += networkCost(expected[i], inputs[i]);
        }

        return cost / inputs.length;
    }

    public void fullBackpropagation(double[] input, double[] expected, double learningRate) {
        this.feed(input);
        HiddenLayer lastLayer = this.layers[this.layers.length-1];
        // Assigns delta values for each Node in the output layer.
        for (int node = 0; node < lastLayer.getNodes().length; node++) {
            Node currentNode = lastLayer.getNodes()[node];

            double error = expected[node] - currentNode.getActivation();
            HiddenLayer previous = this.layers[this.layers.length - 2];


            for (int weight = 0; weight < currentNode.getWeights().length; weight++) {
                Node corespondingNode = previous.getNodes()[weight];
                double delta = corespondingNode.getActivation() * learningRate * error;
                currentNode.setWeight(weight, currentNode.getWeights()[weight] + delta);
            }
        }

        for (int layer = this.layers.length - 2; layer <= 1; layer--) {
            HiddenLayer nextLayer = this.layers[layer + 1];
            HiddenLayer currentLayer = this.layers[layer];
            HiddenLayer previousLayer = this.layers[layer - 1];

            for (int currentLayerNode = 0; currentLayerNode < currentLayer.getNodes().length; currentLayerNode++) {
                double sum = 0;
                for (int nextLayerNode = 0; nextLayerNode < nextLayer.getNodes().length; nextLayerNode++) {
                    double weight = nextLayer.getNodes()[nextLayerNode].getWeights()[currentLayerNode];
                    double delta = nextLayer.getNodes()[nextLayerNode].delta;
                    sum += weight * delta;
                }

                for (int previousLayerNode = 0; previousLayerNode < previousLayer.getNodes().length; previousLayerNode++) {
                    double delta = sum * learningRate * previousLayer.getNodes()[previousLayerNode].getActivation();
                    currentLayer.getNodes()[currentLayerNode].delta += delta;
                }
            }
        }



        

    }
}
