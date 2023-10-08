import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        double targetAccuracy = 0.05;
        int loops = 1000;
        int bipolar = 0;
        int upperLimit = 15000;

        double[][] inputSetsBi = {
                {-1, -1},
                {1, -1},
                {-1, 1},
                {1, 1}
        };
        double[] targetOutputsBi = {-1, 1, 1, -1};

        double[][] inputSets = {
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        };
        double[] targetOutputs = {0, 1, 1, 0};

        NeuralNet neuralNet = new NeuralNet(2);
        neuralNet.initializeWeights();

        double loss;
        double[][] inputs;
        double[] outputs;

        ArrayList<Double> fastLog = new ArrayList<>();
        ArrayList<Double> slowLog = new ArrayList<>();
        ArrayList<Double> currentLog = new ArrayList<>();
        ArrayList<ArrayList<Double>> average = new ArrayList<>();
        ArrayList<Double> plotAverage = new ArrayList<>();
        int slowest = 0;
        int fastest = 10000;

        int averageEpochs = 0;

        if (bipolar == 1) {
            inputs = inputSetsBi;
            outputs = targetOutputsBi;
        } else {
            inputs = inputSets;
            outputs = targetOutputs;
        }

        for (int k = 0; k < loops; k++) {

            currentLog.clear();
            neuralNet.initializeWeights();

            int i = 0;
            int h = 0;
            while (h < upperLimit) {
                if (computeLoss(inputs, outputs, neuralNet) > targetAccuracy) {
                    i++;
                }
                for (int j = 0; j < outputs.length; j++) {
                    neuralNet.train(inputs[j], outputs[j]);
                }

                if (loops == 1) {
                    System.out.println(computeLoss(inputs, outputs, neuralNet));
                }

                currentLog.add(computeLoss(inputs, outputs, neuralNet));
                h++;
            }

            average.add(currentLog);

            if (loops > 1) {
                if (i > slowest) {
                    slowest = i;
                    slowLog = new ArrayList<>(currentLog);
                } if (i < fastest) {
                    fastest = i;
                    fastLog = new ArrayList<>(currentLog);
                }
            }

            averageEpochs += i;
            //System.out.println("No of loop " + (k + 1) + ", No. of epochs " + (i + 1) + ", with total loss " + computeLoss(inputs,outputs, neuralNet));
        }

        averageEpochs = averageEpochs / loops;
        //System.out.println("\nAverage epochs: " + averageEpochs + '\n');
        System.out.println("Fastest: " + fastest + '\n');
        for (double log : fastLog) {
            System.out.println(log);
        }
        System.out.println("\nSlowest: " + slowest + '\n');
        for (double log : slowLog) {
            System.out.println(log);
        }
        System.out.println("\nAverage: " + averageEpochs + '\n');
        double currentAvg = 0;
        for (int j = 0; j < upperLimit; j++) {
            for (int i = 0; i < average.size(); i++) {
                currentAvg += average.get(i).get(j);
            }
            System.out.println(currentAvg/average.size());
            currentAvg = 0;
        }
    }

    private static double computeLoss(double[][] input, double[] output, NeuralNet neuralNet) {
        return neuralNet.loss(input, output);
    }

}
