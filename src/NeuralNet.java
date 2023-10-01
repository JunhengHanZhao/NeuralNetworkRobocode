import Interface.NeuralNetInterface;

import java.io.File;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface {

    private int inputNum;
    private double[] inputs;
    private double [] hiddenOutput;
    private double [] outputWeights;
    private double [][] hiddenWeights;
    private double output;

    private final int hiddenNum = 4;
    private final double learningRate = 0.2;
    private final double momentum = 0;
    private final double errorTarget = 0.05;

    public NeuralNet(int inputNum) {
        this.inputNum = inputNum;

        //Initialize weights to all 0
        zeroWeights();
    }

    public void setOutputWeight(double[] outputWeights) {
        if (outputWeights.length != hiddenNum + 1) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            this.outputWeights = outputWeights;
        }
    }

    public void setHiddenWeight(double[][] hiddenWeights) {
        if (hiddenWeights.length!=hiddenNum || hiddenWeights[0].length!=inputNum+1) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            this.hiddenWeights = hiddenWeights;
        }
    }


    //forward propagation
    @Override
    public double outputFor(double[] X) {
        // check the size of weights
        if (outputWeights.length != hiddenNum + 1 || hiddenWeights.length != hiddenNum || hiddenWeights[0].length != inputNum + 1) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            // begin forward propagation
            this.inputs = X;

            // compute the hidden layer
            double [] hiddenOutput = new double[hiddenNum];
            for (int i = 0; i < hiddenNum; i++) {
                hiddenOutput[i] = hiddenWeights[i][0] * bias;
                for (int j = 1; j < inputNum + 1; j++) {
                    hiddenOutput[i] += hiddenWeights[i][j] * inputs[j-1];
                }
                hiddenOutput[i] = sigmoid(hiddenOutput[i]);
                this.hiddenOutput = hiddenOutput;
            }

            // compute the output layer
            double output = outputWeights[0] * bias;
            for (int i = 1; i < hiddenNum + 1; i++) {
                output += outputWeights[i] * this.hiddenOutput[i-1];
            }
            output = sigmoid(output);
            this.output = output;
            return output;
        }
    }

    @Override
    public double train(double[] X, double argValue) {
        // compute forward propagation result for this training
        double currentOutput = outputFor(X);
        double currentHiddenOutputs[] = hiddenOutput;

        // store current weights for computation
        double[] currentOutputWeights = outputWeights;
        double[][] currentHiddenWeights = hiddenWeights;

        // perform error back propagation
        // the order is reversed because the bottom weights depends on top results
        // set the new weight for output neuron
        double newOutputWeights[] = new double[hiddenNum + 1];
        double outputError = currentOutput * (1 - currentOutput) * (argValue - currentOutput);
        newOutputWeights[0] = currentOutputWeights[0] + learningRate * outputError * bias;
        for (int i = 0; i < hiddenNum; i++) {
            newOutputWeights[i] = currentOutputWeights[i+1] + learningRate * outputError * currentHiddenOutputs[i];
        }

        // set the new weight for hidden neurons
        double newHiddenWeights[][] = new double[hiddenNum][inputNum + 1];
        for (int i = 0; i < hiddenNum; i++) {
            newHiddenWeights[i][0] = currentHiddenWeights [i][0] + learningRate * currentHiddenOutputs[i] * (1 - currentHiddenOutputs[i]) * outputError * newOutputWeights[i] * bias;
            for (int j = 0; j < inputNum; j++) {
                newHiddenWeights[i][j] = currentHiddenWeights[i][j + 1] + learningRate * currentHiddenOutputs[i] * (1 - currentHiddenOutputs[i]) * outputError * newOutputWeights[i] * inputs [j];
            }

        }

        // change the weights in the field
        outputWeights = newOutputWeights;
        hiddenWeights = newHiddenWeights;

        // compute the new output
        return outputFor(inputs);
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
        // set weight for output neuron
        for (int i = 0; i < hiddenNum + 1; i++) {
            outputWeights[i] = Math.random() - 0.5;
        }

        // set weight for hidden neuron
        for (int i = 0; i < hiddenNum; i++) {
            for (int j = 0; j < inputNum + 1; j++) {
                hiddenWeights[i][j] = Math.random() - 0.5;
            }
        }
    }

    @Override
    public void zeroWeights() {
        // set weight for output neuron
        for (int i = 0; i < hiddenNum + 1; i++) {
            outputWeights[i] = 0;
        }

        // set weight for hidden neuron
        for (int i = 0; i < hiddenNum; i++) {
            for (int j = 0; j < inputNum + 1; j++) {
                hiddenWeights[i][j] = 0;
            }
        }
    }
}
