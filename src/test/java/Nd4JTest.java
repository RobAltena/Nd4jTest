import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Nd4jTest {

    @Test
    public void testAppend() {
        INDArray appendTo = Nd4j.ones(DataType.DOUBLE,3, 3);
        INDArray ret = Nd4j.append(appendTo, 3, 1, -1);
        assertArrayEquals(new long[] {3, 6}, ret.shape());

        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray otherAppend = Nd4j.append(linspace, 3, 1.0, -1);
        INDArray assertion = Nd4j.create(new double[][] {{1, 3, 1, 1, 1}, {2, 4, 1, 1, 1}});

        assertEquals(assertion, otherAppend);
    }

    @Test
    public void testPrepend() {
        INDArray appendTo = Nd4j.ones(DataType.DOUBLE, 3, 3);
        INDArray ret = Nd4j.append(appendTo, 3, 1, -1);
        assertArrayEquals(new long[] {3, 6}, ret.shape());

        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{1, 1, 1, 1, 3}, {1, 1, 1, 2, 4}});

        INDArray prepend = Nd4j.prepend(linspace, 3, 1.0, -1);
        assertEquals(assertion, prepend);

    }
}
