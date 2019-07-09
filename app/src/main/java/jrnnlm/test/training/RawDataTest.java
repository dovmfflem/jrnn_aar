package jrnnlm.test.training;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;

import java.io.IOException;
import java.util.Arrays;

public class RawDataTest {

    public static void main(String[] argv) throws IOException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.vocabSize = 50;
        conf.hiddenSize = 2;
        conf.maxIters = 100;
        conf.bpttBlock = 2;
        conf.trainData = new int[]{2, 4, 6, 8};
        conf.validData = new int[]{10, 12, 14, 16};
        RNNLM lm = new RNNLM(conf);

        Arrays.fill(lm.inputSynapse.weights.getData(), 0.1);
        Arrays.fill(lm.recurrentSynapse.weights.getData(), 0.1);
        Arrays.fill(lm.hiddenSynapse.weights.getData(), 0.1);

      
        
        lm.train();

        lm.inputSynapse.weights.print();
        lm.hiddenSynapse.weights.print();
        lm.recurrentSynapse.weights.print();

    }
}
