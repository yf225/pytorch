package org.pytorch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class PytorchInstrumentedTests {

  private static final String TEST_MODULE_ASSET_NAME = "test.pt";

  @Before
  public void setUp() {
    System.loadLibrary("pytorch");
  }

  @Test
  public void testForwardNull() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue input =
        IValue.tensor(Tensor.newByteTensor(new long[] {1}, Tensor.allocateByteBuffer(1)));
    assertTrue(input.isTensor());
    final IValue output = module.forward(input);
    assertTrue(output.isNull());
  }

  @Test
  public void testEqBool() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    for (boolean value : new boolean[] {false, true}) {
      final IValue input = IValue.bool(value);
      assertTrue(input.isBool());
      assertTrue(value == input.getBool());
      final IValue output = module.runMethod("eqBool", input);
      assertTrue(output.isBool());
      assertTrue(value == output.getBool());
    }
  }

  @Test
  public void testEqInt() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    for (long value : new long[] {Long.MIN_VALUE, -1024, -1, 0, 1, 1024, Long.MAX_VALUE}) {
      final IValue input = IValue.long64(value);
      assertTrue(input.isLong());
      assertTrue(value == input.getLong());
      final IValue output = module.runMethod("eqInt", input);
      assertTrue(output.isLong());
      assertTrue(value == output.getLong());
    }
  }

  @Test
  public void testEqFloat() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    double[] values =
        new double[] {
          -Double.MAX_VALUE,
          Double.MAX_VALUE,
          -Double.MIN_VALUE,
          Double.MIN_VALUE,
          -Math.exp(1.d),
          -Math.sqrt(2.d),
          -3.1415f,
          3.1415f,
          -1,
          0,
          1,
        };
    for (double value : values) {
      final IValue input = IValue.double64(value);
      assertTrue(input.isDouble());
      assertTrue(value == input.getDouble());
      final IValue output = module.runMethod("eqFloat", input);
      assertTrue(output.isDouble());
      assertTrue(value == output.getDouble());
    }
  }

  @Test
  public void testEqTensor() throws IOException {
    final long[] inputTensorDims = new long[] {1, 3, 224, 224};
    final long numElements = Tensor.numElements(inputTensorDims);
    final float[] inputTensorData = new float[(int) numElements];
    for (int i = 0; i < numElements; ++i) {
      inputTensorData[i] = i;
    }
    final Tensor inputTensor = Tensor.newFloatTensor(inputTensorDims, inputTensorData);

    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final IValue input = IValue.tensor(inputTensor);
    assertTrue(input.isTensor());
    assertTrue(inputTensor == input.getTensor());
    final IValue output = module.runMethod("eqTensor", input);
    assertTrue(output.isTensor());
    final Tensor outputTensor = output.getTensor();
    assertNotNull(outputTensor);
    assertArrayEquals(inputTensorDims, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    for (int i = 0; i < numElements; i++) {
      assertTrue(inputTensorData[i] == outputData[i]);
    }
  }

  @Test
  public void testEqDictIntKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final Map<Long, IValue> inputMap = new HashMap<>();

    inputMap.put(Long.MIN_VALUE, IValue.long64(-Long.MIN_VALUE));
    inputMap.put(Long.MAX_VALUE, IValue.long64(-Long.MAX_VALUE));
    inputMap.put(0l, IValue.long64(0l));
    inputMap.put(1l, IValue.long64(-1l));
    inputMap.put(-1l, IValue.long64(1l));

    final IValue input = IValue.dictLongKey(inputMap);
    assertTrue(input.isDictLongKey());

    final IValue output = module.runMethod("eqDictIntKeyIntValue", input);
    assertTrue(output.isDictLongKey());

    final Map<Long, IValue> outputMap = output.getDictLongKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<Long, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testEqDictStrKeyIntValue() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    final Map<String, IValue> inputMap = new HashMap<>();

    inputMap.put("long_min_value", IValue.long64(Long.MIN_VALUE));
    inputMap.put("long_max_value", IValue.long64(Long.MAX_VALUE));
    inputMap.put("long_0", IValue.long64(0l));
    inputMap.put("long_1", IValue.long64(1l));
    inputMap.put("long_-1", IValue.long64(-1l));

    final IValue input = IValue.dictStringKey(inputMap);
    assertTrue(input.isDictStringKey());

    final IValue output = module.runMethod("eqDictStrKeyIntValue", input);
    assertTrue(output.isDictStringKey());

    final Map<String, IValue> outputMap = output.getDictStringKey();
    assertTrue(inputMap.size() == outputMap.size());
    for (Map.Entry<String, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).getLong() == entry.getValue().getLong());
    }
  }

  @Test
  public void testListIntSumReturnTuple() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    for (int n : new int[] {0, 1, 128}) {
      long[] a = new long[n];
      long sum = 0;
      for (int i = 0; i < n; i++) {
        a[i] = i;
        sum += a[i];
      }
      final IValue input = IValue.longList(a);
      assertTrue(input.isLongList());

      final IValue output = module.runMethod("listIntSumReturnTuple", input);

      assertTrue(output.isTuple());
      assertTrue(2 == output.getTuple().length);

      IValue output0 = output.getTuple()[0];
      IValue output1 = output.getTuple()[1];

      assertArrayEquals(a, output0.getLongList());
      assertTrue(sum == output1.getLong());
    }
  }

  @Test
  public void testOptionalIntIsNone() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    assertFalse(module.runMethod("optionalIntIsNone", IValue.long64(1l)).getBool());
    assertTrue(module.runMethod("optionalIntIsNone", IValue.optionalNull()).getBool());
  }

  @Test
  public void testIntEq0None() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));

    assertTrue(module.runMethod("intEq0None", IValue.long64(0l)).isNull());
    assertTrue(module.runMethod("intEq0None", IValue.long64(1l)).getLong() == 1l);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testRunUndefinedMethod() throws IOException {
    final Module module = Module.load(assetFilePath(TEST_MODULE_ASSET_NAME));
    module.runMethod("test_undefined_method_throws_exception");
  }

  @Test
  public void testTensorMethods() {
    long[] dims = new long[] {1, 3, 224, 224};
    final int numel = (int) Tensor.numElements(dims);
    int[] ints = new int[numel];
    float[] floats = new float[numel];

    byte[] bytes = new byte[numel];
    for (int i = 0; i < numel; i++) {
      bytes[i] = (byte) ((i % 255) - 128);
      ints[i] = i;
      floats[i] = i / 1000.f;
    }

    Tensor tensorBytes = Tensor.newByteTensor(dims, bytes);
    assertTrue(tensorBytes.isByteTensor());
    assertArrayEquals(bytes, tensorBytes.getDataAsByteArray());

    Tensor tensorInts = Tensor.newIntTensor(dims, ints);
    assertTrue(tensorInts.isIntTensor());
    assertArrayEquals(ints, tensorInts.getDataAsIntArray());

    Tensor tensorFloats = Tensor.newFloatTensor(dims, floats);
    assertTrue(tensorFloats.isFloatTensor());
    float[] floatsOut = tensorFloats.getDataAsFloatArray();
    assertTrue(floatsOut.length == numel);
    for (int i = 0; i < numel; i++) {
      assertTrue(floats[i] == floatsOut[i]);
    }
  }

  @Test(expected = IllegalStateException.class)
  public void testTensorIllegalStateOnWrongType() {
    long[] dims = new long[] {1, 3, 224, 224};
    final int numel = (int) Tensor.numElements(dims);
    float[] floats = new float[numel];
    Tensor tensorFloats = Tensor.newFloatTensor(dims, floats);
    assertTrue(tensorFloats.isFloatTensor());
    tensorFloats.getDataAsByteArray();
  }

  private static String assetFilePath(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    File file = new File(appContext.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = appContext.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      throw e;
    }
  }
}
