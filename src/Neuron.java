public class Neuron {

    private int numInput;
    private double[] inputs;
    private double[] weights;
    private final double bias = 1.0;

    public Neuron(int numInput) {
        this.numInput = numInput;
        weights = new double[numInput + 1]; // +1 for bias input
    }

    public double output(double[] inputVector) {
        if (weights.length != inputVector.length + 1) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            this.inputs = inputVector;
            double weightedSum = 0.0;
            weightedSum = weights[0] * bias; // the bias component, bias has weight 1
            for (int i = 1; i < this.weights.length; i++) {
                weightedSum += weights[i] * inputVector[i - 1];
            }
            return weightedSum;
        }
    }

    public void setWeights(double[] weightVector) {
        if (weights.length != weightVector.length) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            for (int i = 0; i < weightVector.length; i++) {
                weights[i] = weightVector[i];
            }
        }
    }

    public double loss(double[][] trainInputVectors, double[] trainTargetOutputs) {
        double loss = 0.0;
        for (int i = 0; i < trainTargetOutputs.length; i++) {
            double y = this.output(trainInputVectors[i]);
            loss += Math.pow(y - trainTargetOutputs[i], 2);
        }
        return loss;
    }

    public double[] getWeights() {
        return weights;
    }

}
