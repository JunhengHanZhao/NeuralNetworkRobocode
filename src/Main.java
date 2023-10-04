public class Main {
    public static void main(String[] args) {
        double targetAccuracy = 0.05;
        int loops = 10;

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

        NeuralNet train = new NeuralNet(2);

        for (int k = 0; k < loops; k ++) {
            train.initializeWeights();
            int i = 1;
//            while (train.loss(inputSetsBi, targetOutputsBi) > targetAccuracy) {
//                for (int j = 0; j < targetOutputs.length; j++) {
//                    train.train(inputSetsBi[j], targetOutputsBi[j]);
//                    System.out.println("No of loop " + (k + 1) + ", No. of epochs " + i + ", with total loss " + train.loss(inputSets, targetOutputs));
//
//                }
//                i++;
//            }

            while (train.loss(inputSets, targetOutputs) > targetAccuracy) {
                for (int j = 0; j < targetOutputs.length; j++) {
                    train.train(inputSets[j], targetOutputs[j]);
                }
                i++;
            }

            System.out.println("No of loop " + (k + 1) + ", No. of epochs " + i + ", with total loss " + train.loss(inputSets, targetOutputs));
        }
    }
}
