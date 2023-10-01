import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class NeuralNetTest {

    @Test
    //test forward propagation
    public void testNeuralNetOutput(){
        double[] inputVector = {1, -1};
        double[] outputNeuronWeights = {-0.3, 0.3, 0.2, 0.4, 0.1};
        double[][] hiddenNeuronWeights = {
                {0.34, 0.42, -0.11},
                {-0.45, 0.26, 0.1},
                {0.22, -0.3, 0.44},
                {0.35, -0.2, 0.25}
        };

        NeuralNet testNeuralNet = new NeuralNet(2);
        testNeuralNet.setOutputWeight(outputNeuronWeights);
        testNeuralNet.setHiddenWeight(hiddenNeuronWeights);

        assertEquals("0.548", String.format("%.3f",testNeuralNet.outputFor(inputVector)));
    }

}
