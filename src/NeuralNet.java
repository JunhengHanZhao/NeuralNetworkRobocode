import Interface.NeuralNetInterface;

import java.io.File;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface {

    private int inputNum;
    private double[] inputs;
    private double [] hiddenOutput;
    private double [] prevOutputWeights;
    private double [][] prevHiddenWeights;
    private double [] outputWeights;
    private double [][] hiddenWeights;
    private double output;
    private final int bipolar = 1;

    private final int hiddenNum = 4;
    private final double learningRate = 0.2;
    private final double momentum = 0.9;
    private final double errorTarget = 0.05;

    public NeuralNet(int inputNum) {
        this.inputNum = inputNum;

        outputWeights = new double[hiddenNum + 1];
        hiddenOutput = new double[hiddenNum];
        hiddenWeights = new double[hiddenNum][inputNum + 1];


        //Initialize weights to all 0
        zeroWeights();
        initializeWeights();
        prevOutputWeights = outputWeights;
        prevHiddenWeights = hiddenWeights;
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
            for (int i = 0; i < hiddenNum; i++) {
                hiddenOutput[i] = hiddenWeights[i][0] * bias;
                for (int j = 1; j < inputNum + 1; j++) {
                    hiddenOutput[i] += hiddenWeights[i][j] * inputs[j - 1];
                }

                // activation function
                if (bipolar == 1) {
                    hiddenOutput[i] = customSigmoid(hiddenOutput[i]);
                } else {
                    hiddenOutput[i] = sigmoid(hiddenOutput[i]);
                }

            }

            // compute the output layer
            double output = outputWeights[0] * bias;
            for (int i = 1; i < hiddenNum + 1; i++) {
                output += outputWeights[i] * hiddenOutput[i-1];
            }

            // activation function
            if (bipolar == 1) {
                output = customSigmoid(output);
            } else {
                output = sigmoid(output);
            }

            this.output = output;

            return output;
        }
    }

    @Override
    public double train(double[] X, double argValue) {
        // compute forward propagation result for this training
        outputFor(X);

        // perform error back propagation
        // compute the output wight first then the hidden weights because bottom weights depends on top results
        // set the new weight for output neuron
        double newOutputWeights[] = new double[outputWeights.length];

        // computer the  output error
        double outputError;
        if (bipolar == 1) {
            outputError = 0.5 * (1 + output) * (1 - output) * (argValue - output);
        } else {
            outputError = output * (1 - output) * (argValue - output);
        }

        // the part for bias term
        newOutputWeights[0] = outputWeights[0] + momentum * (outputWeights[0] - prevOutputWeights [0]) + learningRate * outputError * bias;
        for (int i = 1; i < hiddenNum + 1; i++) {
            newOutputWeights[i] = outputWeights[i] + momentum * (outputWeights[i] - prevOutputWeights [i]) + learningRate * outputError * hiddenOutput[i-1];
        }

        // set the new weight for hidden neurons
        double newHiddenWeights[][] = new double[hiddenNum][inputNum + 1];
        for (int i = 0; i < hiddenNum; i++) {
            // the part for bias term
            // use the new output weight
            double[] hiddenError = new double[hiddenNum];

            // compute hidden error
            if (bipolar == 1) {
                hiddenError[i] = 0.5 * (1 + hiddenOutput[i]) * (1 - hiddenOutput[i]) * outputError * newOutputWeights[i + 1];
            } else {
                hiddenError[i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * outputError * newOutputWeights[i + 1];
            }

            newHiddenWeights[i][0] = hiddenWeights [i][0] + momentum * (hiddenWeights[i][0] - prevHiddenWeights[i][0]) + learningRate * hiddenError[i] * bias;
            for (int j = 1; j < inputNum + 1; j++) {
                newHiddenWeights[i][j] = hiddenWeights[i][j] + momentum * (hiddenWeights[i][j] - prevHiddenWeights[i][j]) + learningRate * hiddenError[i] * inputs[j-1];
            }

        }

        // change the weights in the field
        prevOutputWeights = outputWeights;
        prevHiddenWeights = hiddenWeights;

        outputWeights = newOutputWeights;
        hiddenWeights = newHiddenWeights;

        // compute the new output
        outputFor(inputs);

        // compute the error
        if (bipolar == 1) {
            return (1 + output) * (1 - output) * (argValue - output);
        } else {
            return output * (1 - output) * (argValue - output);
        }
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
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    //a custom activation function
    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    public double customSigmoid(double x) {
        Integer a = -1;
        Integer b = 1;
        return (b - a) / (1 + Math.exp(-x)) + a;
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

        prevOutputWeights = outputWeights;
        prevHiddenWeights = hiddenWeights;
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

    public double[] getOutputWeights() {
        return outputWeights;
    }

    public double[][] getHiddenWeights() {
        return hiddenWeights;
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

    public double loss(double[][] trainInputVectors, double[] trainTargetOutputs) {
        double loss = 0.0;
        if (trainInputVectors.length != trainTargetOutputs.length) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            for (int i = 0; i < trainTargetOutputs.length; i++) {
                double y = outputFor(trainInputVectors[i]);
                loss += 0.5 * Math.pow(y - trainTargetOutputs[i], 2);
            }
            return loss;
        }
    }

}
