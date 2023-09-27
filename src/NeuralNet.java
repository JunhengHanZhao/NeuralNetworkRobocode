import Interface.NeuralNetInterface;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class NeuralNet implements NeuralNetInterface {

    public Neuron outputNeuron;
    public ArrayList<Neuron> hiddenNeurons;

    private int inputNum;
    private double[] inputs;
    private final int hiddenNum = 4;
    private final double learningRate = 0.2;
    private final double momentum = 0;
    private final double errorTarget = 0.05;
    private double [] outputWeights;
    private double [][] hiddenWeights;

    public NeuralNet(int inputNum) {
        this.inputNum = inputNum;

        // populate hidden layer
        hiddenNeurons = new ArrayList<>();
        for (int i = 0; i < hiddenNum; i++) {
            hiddenNeurons.add(new Neuron(inputNum));
        }

        // the input of output neuron are hidden neurons
        this.outputNeuron = new Neuron(hiddenNum);
    }

    //forward propagation
    @Override
    public double outputFor(double[] X) {
        // check if input vector length match
        if (X.length != inputNum) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            // store the input for debugging
            this.inputs = X;
            // the output array of all hidden neurons
            double[] hiddenOutput = new double[hiddenNum];
            // compute the output of each hidden neuron and populate the array
            for (int i = 0; i < hiddenNeurons.size(); i++) {
                hiddenOutput[i] = sigmoid(hiddenNeurons.get(i).output(X));
            }
            // use hidden neuron output as input
            return sigmoid(outputNeuron.output(hiddenOutput));
        }
    }

    @Override
    public double train(double[] X, double argValue) {
        // compute forward propagation result for this training
        double currentOutput = outputFor(X);

        // store current weights for computation
        double[] currentOutputWeights = outputNeuron.getWeights();
        double[][] currentHiddenWeights = new double [4][inputNum +1];
        for (int i=0; i<hiddenNum; i++) {
            currentHiddenWeights[i] = hiddenNeurons.get(i).getWeights();
        }

        // set the new weight for output neuron


//        double outputNeuronNewWeight[] = new double[hiddenNum + 1];
//        for (int i=0; i<hiddenNum + 1; i++) {
//            outputNeuronNewWeight[i] = outputNeuron.getWeights()[i] + learningRate * currentOutput * (1 - currentOutput) * (argValue - currentOutput);
//        }
//        outputNeuron.setWeights(outputNeuronNewWeight);

        // set the new weight for hidden neurons

    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    @Override
    //activation function
    public double sigmoid(double x) {
        return (double)1 / (1 + Math.exp(-x));
    }

    @Override
    //a custom activation function
    public double customSigmoid(double x) {
        Integer a = -1;
        Integer b = 1;
        return (double)(b - a) / (1 + Math.exp(-x)) + a;
    }

    @Override
    public void initializeWeights() {
        // initialize weight for output neuron
        double[] outputNeuronWeights = new double[hiddenNum +1];
        for (int i = 0; i < hiddenNum + 1; i++) {
            outputNeuronWeights[i] = Math.random() - 0.5;
        }
        outputNeuron.setWeights(outputNeuronWeights);

        // initialize weight for input neuron
        double[][] hiddenNeuronWeights = new double [hiddenNum][inputNum + 1];
        for (int i = 0; i < hiddenNum; i++) {
            for (int j = 0; j < inputNum + 1; j++) {
                hiddenNeuronWeights[i][j] = Math.random() - 0.5;
            }
        }

        for (int k = 0; k < hiddenNeurons.size(); k++) {
            hiddenNeurons.get(k).setWeights(hiddenNeuronWeights[k]);
        }
    }

    @Override
    public void zeroWeights() {
        // set weight for output neuron
        double[] outputNeuronWeights = new double[hiddenNum +1];
        for (int i = 0; i < hiddenNum + 1; i++) {
            outputNeuronWeights[i] = 0;
        }
        outputNeuron.setWeights(outputNeuronWeights);

        // set weight for input neuron
        double[][] hiddenNeuronWeights = new double [hiddenNum][inputNum + 1];
        for (int i = 0; i < hiddenNum; i++) {
            for (int j = 0; j < inputNum + 1; j++) {
                hiddenNeuronWeights[i][j] = 0;
            }
        }

        for (int k = 0; k < hiddenNeurons.size(); k++) {
            hiddenNeurons.get(k).setWeights(hiddenNeuronWeights[k]);
        }
    }
}
