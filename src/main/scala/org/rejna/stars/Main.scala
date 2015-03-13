package org.rejna.stars

import java.nio.FloatBuffer

import scala.util.Random

import com.jogamp.opencl._
import com.jogamp.opencl.CLMemory.Mem._

object Main extends App {
  // set up (uses default CLPlatform and creates context for all devices)
  val context = CLContext.create();
  println("created " + context);

  // always make sure to release the context under all circumstances
  // not needed for this particular sample but recommented
  try {

    // select fastest device
    val device = context.getMaxFlopsDevice()
    println("using " + device)

    // create command queue on device.
    val queue = device.createCommandQueue()

    val elementCount = 1444477 // Length of arrays to process
    val localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256) // Local work size dimensions
    val globalWorkSize = roundUp(localWorkSize, elementCount) // rounded up to the nearest multiple of the localWorkSize

    // load sources, create and build program
    val program = context.createProgram(getClass.getResourceAsStream("/Newton.cl")).build();

    // A, B are input buffers, C is for the result
    val clBufferA = context.createFloatBuffer(globalWorkSize, READ_ONLY);
    val clBufferB = context.createFloatBuffer(globalWorkSize, READ_ONLY);
    val clBufferC = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);

    println("used device memory: "
      + (clBufferA.getCLSize() + clBufferB.getCLSize() + clBufferC.getCLSize()) / 1000000 + "MB");

    // fill input buffers with random numbers
    // (just to have test data; seed is fixed -> results will not change between runs).
    fillBuffer(clBufferA.getBuffer(), 12345);
    fillBuffer(clBufferB.getBuffer(), 67890);

    // get a reference to the kernel function with the name 'VectorAdd'
    // and map the buffers to its input parameters.
    val kernel = program.createCLKernel("VectorAdd");
    kernel.putArgs(clBufferA, clBufferB, clBufferC).putArg(elementCount);

    // asynchronous write of data to GPU device,
    // followed by blocking read to get the computed results back.
    val startTime = System.nanoTime();
    queue.putWriteBuffer(clBufferA, false)
      .putWriteBuffer(clBufferB, false)
      .put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
      .putReadBuffer(clBufferC, true);
    val endTime = System.nanoTime() - startTime;

    // print first few elements of the resulting buffer to the console.
    println("a+b=c results snapshot: ");
    //for(int i = 0; i < 10; i++)
    (0 until 10).foreach { i =>
      print(clBufferC.getBuffer().get() + ", ");
    }
    println("...; " + clBufferC.getBuffer().remaining() + " more");

    println("computation took: " + (endTime / 1000000) + "ms");

  } finally {
    // cleanup all resources associated with this context.
    context.release();
  }

  def fillBuffer(buffer: FloatBuffer, max: Float) = {
    while (buffer.remaining() != 0)
      buffer.put(Random.nextFloat() * max);
    buffer.rewind();
  }

  def roundUp(groupSize: Int, globalSize: Int) =
    globalSize % groupSize match {
      case 0 => globalSize
      case r => globalSize + groupSize - r
    }

}