import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NeuronTest {


    @Test
    public void testOutput() {

        double[] inputVector = {0.5, -0.6};
        double[] testWeights = {1.2, 0.04, -0.96};

        Neuron testNeuron = new Neuron(2);
        testNeuron.setWeights(testWeights);

        double expectedOutput = 1.796;
        double actualOutput = testNeuron.output(inputVector);

        assertEquals(expectedOutput, actualOutput, 0.001);
    }

    @Test
    public void testLoss() {
        double[][] trainingInputVectors = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };

        double[] trainingTargetVector = {
                0.0,
                0.0,
                0.0,
                1.0
        };


        double[] testWeights = {1.2, 0.04, -0.96};
        Neuron testNeuron = new Neuron(2);
        testNeuron.setWeights(testWeights);

        double expectedLoss = 3.5536;
        double actualLoss = testNeuron.loss(trainingInputVectors, trainingTargetVector);

        assertEquals(expectedLoss, actualLoss, 0.01);
    }

}
